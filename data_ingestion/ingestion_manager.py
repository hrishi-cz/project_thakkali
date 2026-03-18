"""
Production-grade async data ingestion with DVC lineage, aiohttp concurrency,
and per-URL fault isolation.
"""

import os
import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import logging

import aiohttp

from data_ingestion.loader import DataLoader

logger = logging.getLogger(__name__)


class DataIngestionManager:
    """
    Async, production-grade data ingestion with:
      - Concurrent multi-URL downloads via aiohttp + asyncio.gather
      - Per-URL fault isolation (404 / conn errors caught, rest continue)
      - SHA-256 cache keying to avoid redundant downloads
      - DVC lineage: `dvc add <cache_path>` called after each successful cache write
    """

    def __init__(self, cache_dir: str = "./data/dataset_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loader = DataLoader()
        self.cache_metadata: Dict[str, Any] = self._load_cache_metadata()

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _load_cache_metadata(self) -> Dict[str, Any]:
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache_metadata(self) -> None:
        metadata_file = self.cache_dir / "cache_metadata.json"
        # Atomic write: temp file + os.replace prevents half-written JSON
        # if the process crashes mid-write or concurrent ingestions race.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.cache_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.cache_metadata, f, indent=2)
            os.replace(tmp_path, str(metadata_file))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _normalize_url(source: str) -> str:
        """
        Canonicalise a URL so that cosmetic variants produce the same hash.

        Normalisation steps:
          1. Strip leading/trailing whitespace
          2. Remove trailing slashes
          3. Lowercase the scheme + host (path is case-sensitive)
          4. Remove ``www.`` prefix from host
          5. Strip query params and fragments (they don't affect the dataset)
          6. Kaggle shortcut: ``kaggle.com/datasets/<owner>/<name>`` is the
             canonical form regardless of full-URL embellishments

        Non-URL strings (local paths) are returned as-is after stripping.
        """
        source = source.strip()
        if not source.startswith(("http://", "https://")):
            # Local path — normalise slashes but keep case (Windows paths)
            return source.replace("\\", "/").rstrip("/")
        parsed = urlparse(source)
        host = parsed.hostname or ""
        host = host.lower().removeprefix("www.")
        path = parsed.path.rstrip("/")
        # Kaggle canonical form: just owner/dataset from the path
        if "kaggle.com" in host and "/datasets/" in path:
            parts = path.split("/datasets/", 1)
            if len(parts) == 2:
                dataset_slug = parts[1].strip("/")
                return f"kaggle://datasets/{dataset_slug}"
        # HuggingFace canonical form: hf://datasets/{owner}/{name}
        if "huggingface.co" in host and "/datasets/" in path:
            parts = path.split("/datasets/", 1)
            if len(parts) == 2:
                dataset_slug = parts[1].strip("/")
                # Strip sub-paths like /viewer, /tree/main, etc.
                slug_parts = dataset_slug.split("/")
                if len(slug_parts) >= 2:
                    dataset_slug = f"{slug_parts[0]}/{slug_parts[1]}"
                return f"hf://datasets/{dataset_slug}"
        return f"{parsed.scheme}://{host}{path}"

    def _generate_hash(self, source: str) -> str:
        """Generate a 16-char SHA-256 hex digest for a normalised source identifier."""
        normalised = self._normalize_url(source)
        return hashlib.sha256(normalised.encode()).hexdigest()[:16]

    def _legacy_hash(self, source: str) -> str:
        """Hash using the raw source string (pre-normalisation era)."""
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    @staticmethod
    def _is_kaggle_url(url: str) -> bool:
        return "kaggle.com/datasets" in url

    @staticmethod
    def _is_huggingface_url(url: str) -> bool:
        return "huggingface.co/datasets" in url

    # ------------------------------------------------------------------ #
    # DVC integration
    # ------------------------------------------------------------------ #

    def _dvc_add(self, cache_path: Path) -> None:
        """
        Register *cache_path* with DVC for data lineage via `dvc add`.
        Failures are logged (not raised) so they never block ingestion.
        """
        try:
            result = subprocess.run(
                ["dvc", "add", str(cache_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(
                    "dvc add failed for %s: %s", cache_path, result.stderr.strip()
                )
            else:
                logger.info("DVC: tracked %s", cache_path)
        except FileNotFoundError:
            logger.info("DVC not installed – skipping lineage for %s", cache_path)
        except Exception as exc:
            logger.warning("dvc add error: %s", exc)

    # ------------------------------------------------------------------ #
    # Public async API
    # ------------------------------------------------------------------ #

    async def ingest_data(
        self,
        sources: Union[str, List[str]],
        force_download: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Concurrently ingest datasets from multiple sources.

        Args:
            sources:        Single URL/path or list of URLs/paths.
            force_download: Force re-download even when cached.

        Returns:
            Tuple:
              - lazy_datasets : {source_hash -> LazyFrame | LazyImageDataset}
              - metadata      : ingestion metadata dict (hashes, failures, timing)
        """
        if isinstance(sources, str):
            sources = [sources]

        metadata: Dict[str, Any] = {
            "sources": sources,
            "ingestion_time": datetime.now().isoformat(),
            "cached_hashes": {},
            "cache_status": {},
            "failed": {},
        }

        tasks = [self._ingest_single(source, force_download) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        lazy_datasets: Dict[str, Any] = {}
        for source, result in zip(sources, results):
            source_hash = self._generate_hash(source)
            if isinstance(result, Exception):
                metadata["failed"][source] = str(result)
                logger.error("Ingestion failed for [%s]: %s", source, result)
            elif result is None:
                metadata["failed"][source] = "Ingestion returned no data"
                logger.error("Ingestion returned None for [%s]", source)
            else:
                lazy_ref, cache_path, from_cache = result
                lazy_datasets[source_hash] = lazy_ref
                metadata["cached_hashes"][source] = source_hash
                metadata["cache_status"][source] = "cached" if from_cache else "downloaded"

        return lazy_datasets, metadata

    # ------------------------------------------------------------------ #
    # Per-source dispatch
    # ------------------------------------------------------------------ #

    async def _ingest_single(
        self,
        source: str,
        force_download: bool,
    ) -> Optional[Tuple[Any, Path, bool]]:
        """
        Ingest one source.  Returns ``(lazy_ref, cache_path, from_cache)``
        or raises.  ``from_cache`` is ``True`` when the dataset was served
        from the local cache without downloading.

        Exceptions propagate to asyncio.gather for per-URL isolation.

        Backward compatibility: if the normalised hash produces a cache miss,
        falls back to the legacy (raw-string) hash.  If found under the legacy
        key the metadata is migrated to the new normalised key in-place.
        """
        source_hash = self._generate_hash(source)
        cache_path = self.cache_dir / source_hash

        # Reload metadata from disk in case another process updated the cache
        self.cache_metadata = self._load_cache_metadata()

        # Cache hit: try normalised hash first, then legacy raw-string hash
        if not force_download:
            # Try normalised hash
            if source_hash in self.cache_metadata:
                lazy_ref = self._loader.load_cached(cache_path)
                if lazy_ref is not None:
                    logger.info("Cache HIT  [%s] -> %s", source_hash, source)
                    return lazy_ref, cache_path, True

            # Try legacy hash (raw string, pre-normalisation era)
            legacy_hash = self._legacy_hash(source)
            if legacy_hash != source_hash and legacy_hash in self.cache_metadata:
                legacy_path = self.cache_dir / legacy_hash
                lazy_ref = self._loader.load_cached(legacy_path)
                if lazy_ref is not None:
                    logger.info(
                        "Cache HIT  [%s] (legacy hash for %s) — migrating to [%s]",
                        legacy_hash, source, source_hash,
                    )
                    # Migrate metadata to normalised key
                    self.cache_metadata[source_hash] = self.cache_metadata.pop(legacy_hash)
                    self.cache_metadata[source_hash]["source_hash"] = source_hash
                    self._save_cache_metadata()
                    # Rename directory so future lookups use the normalised hash
                    if legacy_path.exists() and not cache_path.exists():
                        legacy_path.rename(cache_path)
                    return lazy_ref, cache_path if cache_path.exists() else legacy_path, True

        logger.info("Cache MISS [%s] -> downloading %s", source_hash, source)

        # Route to the right downloader
        if self._is_kaggle_url(source):
            cache_path = await self._ingest_kaggle(source, cache_path)
        elif self._is_huggingface_url(source):
            cache_path = await self._ingest_huggingface(source, cache_path)
        elif source.startswith(("http://", "https://")):
            cache_path = await self._ingest_remote_url(source, cache_path)
        else:
            cache_path = await self._ingest_local_path(source, cache_path)

        # DVC lineage tracking (offloaded to thread – subprocess.run is blocking)
        await asyncio.to_thread(self._dvc_add, cache_path)

        # Persist cache metadata (offloaded – json.dump is blocking I/O)
        await asyncio.to_thread(
            self._update_cache_metadata, source_hash, source, cache_path
        )

        lazy_ref = self._loader.load_cached(cache_path)
        if lazy_ref is None:
            raise RuntimeError(
                f"Cache directory {cache_path} has no recognised data files "
                "after ingestion."
            )
        return lazy_ref, cache_path, False

    # ------------------------------------------------------------------ #
    # Remote URL downloader (aiohttp – truly async)
    # ------------------------------------------------------------------ #

    async def _ingest_remote_url(self, url: str, cache_path: Path) -> Path:
        """
        Download a remote URL with aiohttp.
        Raises FileNotFoundError on 404; ConnectionError on other HTTP errors.
        """
        if "mendeley.com/datasets" in url:
            raise ValueError(
                "Mendeley datasets must be downloaded manually. "
                "Use the local-file upload option instead."
            )

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "data.csv"
        if "." not in filename:
            filename = "data.csv"

        cache_path.mkdir(parents=True, exist_ok=True)
        filepath = cache_path / filename

        # total=7200 (2 h) accommodates 50 GB+ downloads on slower links;
        # sock_read=300 still aborts genuinely stalled connections quickly.
        timeout = aiohttp.ClientTimeout(total=7200, sock_read=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 404:
                    raise FileNotFoundError(f"HTTP 404 Not Found: {url}")
                if response.status != 200:
                    raise ConnectionError(
                        f"HTTP {response.status} while downloading {url}"
                    )
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    raise ValueError(
                        f"URL returns an HTML page, not a direct data file: {url}"
                    )
                # Write chunks via asyncio.to_thread so synchronous disk I/O
                # never blocks the event loop (critical for 50 GB+ files).
                fh = await asyncio.to_thread(open, filepath, "wb")
                try:
                    async for chunk in response.content.iter_chunked(65536):
                        await asyncio.to_thread(fh.write, chunk)
                finally:
                    await asyncio.to_thread(fh.close)

        logger.info("Downloaded %s -> %s", url, filepath)
        return cache_path

    # ------------------------------------------------------------------ #
    # Kaggle downloader (sync, run in thread-pool executor)
    # ------------------------------------------------------------------ #

    async def _ingest_kaggle(self, url: str, cache_path: Path) -> Path:
        """Offload blocking Kaggle CLI call to a thread-pool executor."""
        return await asyncio.to_thread(
            self._ingest_kaggle_sync, url, cache_path
        )

    def _ingest_kaggle_sync(self, url: str, cache_path: Path) -> Path:
        """
        Synchronous Kaggle ingestion.
        Credentials are read from environment variables first, then from ~/.kaggle/kaggle.json as fallback.
        """
        kaggle_username: Optional[str] = os.getenv("KAGGLE_USERNAME")
        kaggle_key: Optional[str] = os.getenv("KAGGLE_KEY")
        
        # Fallback: read from ~/.kaggle/kaggle.json if env vars not set
        if not kaggle_username or not kaggle_key:
            kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_json_path.exists():
                try:
                    with open(kaggle_json_path, "r") as f:
                        creds = json.load(f)
                        kaggle_username = creds.get("username")
                        kaggle_key = creds.get("key")
                        logger.info("Loaded Kaggle credentials from ~/.kaggle/kaggle.json")
                except Exception as e:
                    logger.warning("Failed to read kaggle.json: %s", e)
        
        if not kaggle_username or not kaggle_key:
            raise EnvironmentError(
                "Kaggle credentials missing. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables or ensure ~/.kaggle/kaggle.json exists."
            )

        parts = url.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid Kaggle URL format: {url}")
        dataset_id = f"{parts[-2]}/{parts[-1]}"

        temp_dir = Path(tempfile.gettempdir()) / f"kaggle_{parts[-1]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            env = {
                **os.environ,
                "KAGGLE_USERNAME": kaggle_username,
                "KAGGLE_KEY": kaggle_key,
            }
            result = subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", dataset_id,
                    "-p", str(temp_dir),
                ],
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            if result.returncode != 0:
                if "401" in result.stderr or "Unauthorized" in result.stderr:
                    raise PermissionError(
                        "Kaggle API authentication failed. Check credentials."
                    )
                raise RuntimeError(f"kaggle CLI failed: {result.stderr}")

            zip_files = list(temp_dir.glob("*.zip"))
            if not zip_files:
                raise RuntimeError(
                    f"Expected a zip from Kaggle for {dataset_id}, got none."
                )
            for zf in zip_files:
                with zipfile.ZipFile(zf, "r") as zref:
                    # ZipSlip protection: reject members that escape temp_dir
                    for member in zref.namelist():
                        member_path = (temp_dir / member).resolve()
                        if not str(member_path).startswith(str(temp_dir.resolve())):
                            raise ValueError(
                                f"ZipSlip detected: '{member}' escapes target directory"
                            )
                    zref.extractall(temp_dir)
                zf.unlink()

            csv_files = list(temp_dir.rglob("*.csv"))
            if not csv_files:
                raise RuntimeError(
                    f"No CSV found in Kaggle archive for {dataset_id}."
                )

            cache_path.mkdir(parents=True, exist_ok=True)
            dest = cache_path / csv_files[0].name
            shutil.copy2(csv_files[0], dest)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("Kaggle dataset %s cached at %s", dataset_id, cache_path)
        return cache_path

    # ------------------------------------------------------------------ #
    # HuggingFace downloader (sync, run in thread-pool executor)
    # ------------------------------------------------------------------ #

    async def _ingest_huggingface(self, url: str, cache_path: Path) -> Path:
        """Offload blocking HuggingFace datasets download to a thread."""
        return await asyncio.to_thread(
            self._ingest_huggingface_sync, url, cache_path
        )

    def _ingest_huggingface_sync(self, url: str, cache_path: Path) -> Path:
        """Download a HuggingFace dataset and export to Parquet."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace ingestion. "
                "Install it with: pip install datasets"
            )

        # Extract dataset ID: "owner/name" from URL path
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        # path looks like: "datasets/owner/name" or "datasets/owner/name/..."
        parts = path.split("/")
        if len(parts) < 3 or parts[0] != "datasets":
            raise ValueError(
                f"Cannot parse HuggingFace dataset ID from URL: {url}"
            )
        dataset_id = f"{parts[1]}/{parts[2]}"

        logger.info("Loading HuggingFace dataset: %s", dataset_id)
        try:
            ds = load_dataset(dataset_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HuggingFace dataset '{dataset_id}': {exc}"
            ) from exc

        # Export the best split to Parquet in cache_path
        cache_path.mkdir(parents=True, exist_ok=True)

        # Pick the most useful split: train > test > validation > first available
        if hasattr(ds, "keys"):
            # DatasetDict with splits
            for preferred in ("train", "test", "validation"):
                if preferred in ds:
                    split_ds = ds[preferred]
                    break
            else:
                split_ds = ds[next(iter(ds))]
        else:
            # Single Dataset (no splits)
            split_ds = ds

        out_path = cache_path / "data.parquet"
        split_ds.to_parquet(str(out_path))
        logger.info(
            "HuggingFace dataset %s (%d rows) cached at %s",
            dataset_id, len(split_ds), out_path,
        )
        return cache_path

    # ------------------------------------------------------------------ #
    # Local path (run in executor to avoid blocking the event loop)
    # ------------------------------------------------------------------ #

    async def _ingest_local_path(self, path: str, cache_path: Path) -> Path:
        return await asyncio.to_thread(
            self._copy_local_path, path, cache_path
        )

    def _copy_local_path(self, path: str, cache_path: Path) -> Path:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Local file not found: {path}")
        cache_path.mkdir(parents=True, exist_ok=True)
        dest = cache_path / src.name
        shutil.copy2(src, dest)
        logger.info("Local file copied %s -> %s", src, dest)
        return cache_path

    # ------------------------------------------------------------------ #
    # Cache metadata persistence
    # ------------------------------------------------------------------ #

    def _update_cache_metadata(
        self,
        source_hash: str,
        source: str,
        cache_path: Path,
    ) -> None:
        meta: Dict[str, Any] = {
            "source": source,
            "source_hash": source_hash,
            "timestamp": datetime.now().isoformat(),
            "cache_path": str(cache_path),
        }
        data_files = list(cache_path.glob("*.parquet")) + list(cache_path.glob("*.csv"))
        if data_files:
            meta["size_mb"] = round(os.path.getsize(data_files[0]) / (1024 * 1024), 3)
        self.cache_metadata[source_hash] = meta
        self._save_cache_metadata()

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            "total_cached": len(self.cache_metadata),
            "cache_dir": str(self.cache_dir),
            "cached_items": list(self.cache_metadata.keys()),
            "metadata": self.cache_metadata,
        }

    def clear_cache(self, source_hash: Optional[str] = None) -> None:
        if source_hash:
            cache_path = self.cache_dir / source_hash
            if cache_path.exists():
                shutil.rmtree(cache_path)
            self.cache_metadata.pop(source_hash, None)
            self._save_cache_metadata()
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {}
            self._save_cache_metadata()
