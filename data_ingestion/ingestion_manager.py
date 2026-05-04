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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import logging

import aiohttp

from data_ingestion.loader import DataLoader
from data_ingestion.dataset_object import DatasetObject
from data_ingestion.sampling import validate_dataset

logger = logging.getLogger(__name__)


class DataIngestionManager:
    """
    Async, production-grade data ingestion with:
      - Concurrent multi-URL downloads via aiohttp + asyncio.gather
      - Per-URL fault isolation (404 / conn errors caught, rest continue)
      - SHA-256 cache keying to avoid redundant downloads
      - DVC lineage: `dvc add <cache_path>` called after each successful cache write
    """

    def __init__(self, cache_dir: Union[str, Path] = "./data/dataset_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Mapping of str session ID -> list of successfully ingested hashes
        self.session_datasets: Dict[str, List[str]] = {}
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
        source = source.strip().strip("\"'")  # remove surrounding quotes users may copy-paste
        if not source.startswith(("http://", "https://")):
            # Local path — normalise slashes but keep case (Windows paths).
            # Handle double-paste: Windows "Copy as path" produces `"C:\path"`;
            # if pasted twice the result is `C:\path" "C:\path` after outer strip.
            # Detect by looking for `" "` or repeated-path patterns and take the first segment.
            _norm = source.replace("\\", "/").rstrip("/")
            if '" ' in _norm or "' " in _norm:
                # Take only the first path segment (before the first space-quote boundary)
                _norm = _norm.split('"')[0].split("'")[0].rstrip()
            return _norm
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
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[Dict[str, DatasetObject], Dict[str, Any]]:
        """
        Concurrently ingest datasets from multiple sources.

        Args:
            sources:        Single URL/path or list of URLs/paths.
            force_download: Force re-download even when cached.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple:
              - datasets      : {source_hash -> DatasetObject} (standardized, validated)
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

        tasks = [self._ingest_single(source, force_download, progress_callback) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        datasets: Dict[str, DatasetObject] = {}
        for source, result in zip(sources, results):
            source_hash = self._generate_hash(source)
            if isinstance(result, Exception):
                metadata["failed"][source] = str(result)
                logger.error("Ingestion failed for [%s]: %s", source, result)
            elif result is None:
                metadata["failed"][source] = "Ingestion returned no data"
                logger.error("Ingestion returned None for [%s]", source)
            else:
                lazy_ref, cache_path = result
                
                # Wrap in DatasetObject with metadata
                ingestion_meta = self.cache_metadata.get(
                    source_hash,
                    {"source": source, "cache_path": str(cache_path)},
                )
                dataset_obj = DatasetObject(
                    dataset_id=source_hash,
                    lazy_data=lazy_ref,
                    metadata=ingestion_meta,
                )
                
                # Validate before adding to results
                try:
                    validate_dataset(dataset_obj)
                    datasets[source_hash] = dataset_obj
                    metadata["cached_hashes"][source] = source_hash
                    metadata["cache_status"][source] = "ok"
                    logger.info("Dataset [%s] validated and ready", source_hash)
                except ValueError as ve:
                    metadata["failed"][source] = str(ve)
                    logger.error(
                        "Dataset validation failed for [%s]: %s", source_hash, ve
                    )

        return datasets, metadata

    # ------------------------------------------------------------------ #
    # Per-source dispatch
    # ------------------------------------------------------------------ #

    async def _ingest_single(
        self,
        source: str,
        force_download: bool,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Optional[Tuple[Any, Path]]:
        """
        Ingest one source.  Returns (lazy_ref, cache_path) tuple or raises.
        Exceptions propagate to asyncio.gather for per-URL isolation.

        Backward compatibility: if the normalised hash produces a cache miss,
        falls back to the legacy (raw-string) hash.  If found under the legacy
        key the metadata is migrated to the new normalised key in-place.
        
        Note: Returns raw lazy_ref here (not DatasetObject). Wrapping happens
        in ingest_data() after all ingestion is complete.
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
                    return lazy_ref, cache_path

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
                    return lazy_ref, cache_path if cache_path.exists() else legacy_path

        logger.info("Cache MISS [%s] -> downloading %s", source_hash, source)

        # Route to the right downloader
        if self._is_kaggle_url(source):
            cache_path = await self._ingest_kaggle(source, cache_path, progress_callback)
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
            # Diagnose: list what was actually found so the user knows what's missing
            _found_exts: set = set()
            if cache_path.exists():
                for _f in cache_path.rglob("*"):
                    if _f.is_file() and _f.suffix:
                        _found_exts.add(_f.suffix.lower())
            _ext_hint = (
                f" Found extensions: {sorted(_found_exts)}" if _found_exts
                else " Cache directory is empty."
            )
            raise RuntimeError(
                f"Cache directory {cache_path} has no recognised data files "
                f"after ingestion.{_ext_hint} "
                "Recognised: .parquet, .csv, .xlsx, .jsonl, .jpg/.png/.jpeg (images)."
            )
        return lazy_ref, cache_path

    # ------------------------------------------------------------------ #
    # Remote URL downloader (aiohttp – truly async)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hf_token() -> str:
        """Return HuggingFace API token from env, or empty string."""
        return os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")

    @staticmethod
    def _resolve_huggingface_url(url: str) -> str:
        """Convert a HuggingFace dataset page URL to a direct parquet/data file URL.

        Resolution strategy (in order):
          1. HF datasets-server parquet API  (works for most public datasets)
          2. HF Hub repo files API           (fallback for datasets not indexed by datasets-server)

        For gated datasets that require terms acceptance, raises ValueError with
        actionable instructions.
        """
        import urllib.request as _req
        import urllib.error as _uerr
        import json as _json
        from urllib.parse import urlparse as _up

        parsed = _up(url)
        if "huggingface.co" not in (parsed.hostname or ""):
            return url
        parts = [p for p in parsed.path.split("/") if p]
        if not (len(parts) >= 3 and parts[0] == "datasets"):
            return url

        owner, name = parts[1], parts[2]
        token = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")

        def _authed_request(api_url: str) -> "_req.Request":
            r = _req.Request(api_url)
            if token:
                r.add_header("Authorization", f"Bearer {token}")
            return r

        # ── Strategy 1: datasets-server parquet index ──────────────────────
        try:
            api = f"https://datasets-server.huggingface.co/parquet?dataset={owner}/{name}"
            with _req.urlopen(_authed_request(api), timeout=15) as r:
                data = _json.loads(r.read())
            parquet_files = data.get("parquet_files", [])
            for split_pref in ("train", "validation", "test"):
                for f in parquet_files:
                    if f.get("split") == split_pref:
                        direct = f.get("url", "")
                        if direct:
                            logger.info("HF datasets-server resolved: %s -> %s", url, direct)
                            return direct
            if parquet_files:
                return parquet_files[0].get("url", url)
        except _uerr.HTTPError as exc:
            if exc.code == 401:
                raise ValueError(
                    f"'{owner}/{name}' requires authentication AND accepted dataset terms. "
                    "1) Set HF_TOKEN in .env  2) Visit https://huggingface.co/datasets/"
                    f"{owner}/{name} and click 'Access repository' to accept the terms, "
                    "then restart the API."
                ) from exc
            # 404 = dataset not indexed by datasets-server; try Hub files API next
            logger.debug("datasets-server 404 for %s/%s, trying Hub files API", owner, name)
        except Exception as exc:
            logger.debug("datasets-server failed for %s/%s: %s", owner, name, exc)

        # ── Strategy 2: HF Hub repo files API (finds raw data files) ──────
        try:
            api2 = f"https://huggingface.co/api/datasets/{owner}/{name}"
            with _req.urlopen(_authed_request(api2), timeout=15) as r:
                meta = _json.loads(r.read())
            # Prefer parquet siblings, then csv, then jsonl
            siblings = meta.get("siblings", [])
            for ext in (".parquet", ".csv", ".jsonl", ".json"):
                for s in siblings:
                    rfilename = s.get("rfilename", "")
                    if rfilename.endswith(ext) and "train" in rfilename.lower():
                        direct = f"https://huggingface.co/datasets/{owner}/{name}/resolve/main/{rfilename}"
                        logger.info("HF Hub files API resolved: %s -> %s", url, direct)
                        return direct
                # Second pass: any file of that ext
                for s in siblings:
                    rfilename = s.get("rfilename", "")
                    if rfilename.endswith(ext):
                        direct = f"https://huggingface.co/datasets/{owner}/{name}/resolve/main/{rfilename}"
                        logger.info("HF Hub files API resolved (non-train): %s -> %s", url, direct)
                        return direct
        except _uerr.HTTPError as exc:
            if exc.code == 401:
                raise ValueError(
                    f"'{owner}/{name}' requires authentication AND accepted dataset terms. "
                    "1) Set HF_TOKEN in .env  2) Visit https://huggingface.co/datasets/"
                    f"{owner}/{name} and click 'Access repository' to accept the terms, "
                    "then restart the API."
                ) from exc
            logger.warning("HF Hub API failed for %s/%s: %s", owner, name, exc)
        except Exception as exc:
            logger.warning("HF Hub API failed for %s/%s: %s", owner, name, exc)

        return url

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

        # Resolve HuggingFace dataset page URLs to direct parquet file URLs
        if "huggingface.co/datasets/" in url and not url.endswith((".parquet", ".csv", ".json", ".jsonl")):
            url = self._resolve_huggingface_url(url)

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "data.csv"
        if "." not in filename:
            filename = "data.csv"

        cache_path.mkdir(parents=True, exist_ok=True)
        filepath = cache_path / filename

        # total=7200 (2 h) accommodates 50 GB+ downloads on slower links;
        # sock_read=300 still aborts genuinely stalled connections quickly.
        timeout = aiohttp.ClientTimeout(total=7200, sock_read=300)
        _hf_token = self._hf_token()
        _headers = {"Authorization": f"Bearer {_hf_token}"} if _hf_token and "huggingface.co" in url else {}
        async with aiohttp.ClientSession(timeout=timeout, headers=_headers) as session:
            async with session.get(url) as response:
                if response.status == 404:
                    raise FileNotFoundError(f"HTTP 404 Not Found: {url}")
                if response.status == 401:
                    _ds_hint = ""
                    if "huggingface.co" in url:
                        _ds_hint = (
                            " This dataset is private or gated. Options: "
                            "(1) Visit the dataset page on huggingface.co and click 'Access repository' to accept terms, "
                            "(2) Use a local fixture file instead (e.g. data/fixtures/mmimdb_smoke.csv for MMIMDB)."
                        )
                    raise ValueError(f"HTTP 401 Unauthorized.{_ds_hint}")
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

    async def _ingest_kaggle(self, url: str, cache_path: Path, progress_callback: Optional[Callable[[int, str], None]] = None) -> Path:
        """Offload blocking Kaggle CLI call to a thread-pool executor."""
        return await asyncio.to_thread(
            self._ingest_kaggle_sync, url, cache_path, progress_callback
        )

    def _ingest_kaggle_sync(self, url: str, cache_path: Path, progress_callback: Optional[Callable[[int, str], None]] = None) -> Path:
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
            proc = subprocess.Popen(
                [
                    "kaggle", "datasets", "download",
                    "-d", dataset_id,
                    "-p", str(temp_dir),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # Binary mode so we can split on \r (tqdm carriage-return progress)
                text=False,
                env=env,
            )

            error_output = ""
            _buf = b""
            for raw_chunk in iter(lambda: proc.stdout.read(512), b""):
                _buf += raw_chunk
                # Split on both \r and \n — tqdm uses \r to overwrite lines
                parts = _buf.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
                _buf = parts[-1]  # incomplete last chunk — keep for next iteration
                for raw_line in parts[:-1]:
                    line = raw_line.decode("utf-8", errors="replace")
                    error_output += line + "\n"
                    if "%" in line and progress_callback is not None:
                        try:
                            # tqdm line: "  23%|██       | 450M/1.87G [01:45<...]"
                            pct_str = line.strip().split("%")[0].split()[-1].strip()
                            pct = int(float(pct_str))
                            if 0 < pct <= 100:
                                progress_callback(pct, f"Downloading {dataset_id}... {pct}%")
                        except Exception:
                            pass

            # Flush remaining buffer
            if _buf:
                error_output += _buf.decode("utf-8", errors="replace")

            proc.stdout.close()
            returncode = proc.wait(timeout=1800)

            if returncode != 0:
                if "401" in error_output or "Unauthorized" in error_output:
                    raise PermissionError(
                        "Kaggle API authentication failed. Check credentials."
                    )
                raise RuntimeError(f"kaggle CLI failed: {error_output}")

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
            csv_files = list(temp_dir.rglob("*.csv"))
            image_files = list(temp_dir.rglob("*.jpg")) + list(temp_dir.rglob("*.jpeg")) + list(temp_dir.rglob("*.png"))
            
            if not csv_files and not image_files:
                raise RuntimeError(
                    f"No CSV or Images found in Kaggle archive for {dataset_id}."
                )
                
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Universal Multimodal Extraction: move EVERYTHING from the Kaggle Zip directly to the cache folder!
            for item in temp_dir.iterdir():
                dest = cache_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(cache_path))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("Kaggle dataset %s cached at %s", dataset_id, cache_path)
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

        # Zip archives: extract contents directly into the cache directory
        # so load_cached can discover csv/parquet/images inside.
        if src.suffix.lower() == ".zip":
            with zipfile.ZipFile(src, "r") as zref:
                # ZipSlip protection
                for member in zref.namelist():
                    member_path = (cache_path / member).resolve()
                    if not str(member_path).startswith(str(cache_path.resolve())):
                        raise ValueError(f"ZipSlip detected: '{member}' escapes target directory")
                zref.extractall(cache_path)
            logger.info("Local zip extracted %s -> %s", src, cache_path)
            return cache_path

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
        data_files = (
            list(cache_path.glob("*.parquet"))
            + list(cache_path.glob("*.csv"))
            + list(cache_path.glob("*.xlsx"))
            + list(cache_path.glob("*.xls"))
        )
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
