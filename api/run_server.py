"""
APEX Framework – Production Process Manager & Watchdog.

Launches the FastAPI backend (Uvicorn) and Streamlit frontend as supervised
child processes with real-time log streaming, rotating file retention, health
monitoring, and graceful teardown.

Usage
-----
    python run_server.py

Endpoints
---------
    API:  http://localhost:8001
    UI:   http://localhost:8501

Logs
----
    logs/server_audit.log   (RotatingFileHandler, 10 MB x 5 backups)

Signals
-------
    Ctrl+C / SIGTERM  → graceful shutdown of both processes.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent  # project root (one level above api/)

API_HOST: str = "0.0.0.0"
API_PORT: int = 8001
UI_PORT: int = 8501

LOG_DIR: Path = Path(__file__).resolve().parent / "logs"  # api/logs
LOG_FILE: Path = LOG_DIR / "server_audit.log"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5

SHUTDOWN_GRACE_SECONDS: float = 3.0
WATCHDOG_POLL_INTERVAL: float = 0.5

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> logging.Logger:
    """
    Build a logger that writes to both stderr (coloured prefixes) and a
    rotating file with precise ISO-8601 timestamps.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("apex.server")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log: logging.Logger = _configure_logging()

# ---------------------------------------------------------------------------
# Managed process descriptor
# ---------------------------------------------------------------------------


class ManagedProcess:
    """
    Wraps a ``subprocess.Popen`` child with a human-readable tag, a
    background reader thread for each stdio stream, and helpers for
    graceful / forced termination.
    """

    def __init__(self, tag: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
        self.tag: str = tag
        self.cmd: List[str] = cmd
        self.env: Optional[Dict[str, str]] = env
        self.proc: Optional[subprocess.Popen[str]] = None
        self._readers: List[threading.Thread] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Spawn the child process and attach stdout/stderr reader threads."""
        log.info("[%s] Starting: %s", self.tag, " ".join(self.cmd))

        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,                       # line-buffered
            env=self.env,
            cwd=str(ROOT_DIR),
            # On Windows, CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32" else 0
            ),
        )

        for stream, level in [
            (self.proc.stdout, logging.INFO),
            (self.proc.stderr, logging.WARNING),
        ]:
            if stream is not None:
                t = threading.Thread(
                    target=self._read_stream,
                    args=(stream, level),
                    daemon=True,
                )
                t.start()
                self._readers.append(t)

        log.info("[%s] PID %d started", self.tag, self.proc.pid)

    def poll(self) -> Optional[int]:
        """Non-blocking check.  Returns exit code or ``None`` if still running."""
        if self.proc is None:
            return -1
        return self.proc.poll()

    def terminate(self) -> None:
        """Send a polite termination signal."""
        if self.proc is None or self.proc.poll() is not None:
            return
        log.info("[%s] Sending TERMINATE to PID %d", self.tag, self.proc.pid)
        try:
            self.proc.terminate()
        except OSError as exc:
            log.warning("[%s] terminate() failed: %s", self.tag, exc)

    def kill(self) -> None:
        """Forcefully kill the child."""
        if self.proc is None or self.proc.poll() is not None:
            return
        log.warning("[%s] Sending KILL to PID %d", self.tag, self.proc.pid)
        try:
            self.proc.kill()
        except OSError as exc:
            log.warning("[%s] kill() failed: %s", self.tag, exc)

    def wait(self, timeout: float = 5.0) -> Optional[int]:
        """Block until the process exits or *timeout* seconds elapse."""
        if self.proc is None:
            return -1
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    @property
    def pid(self) -> int:
        return self.proc.pid if self.proc else -1

    # ------------------------------------------------------------------ #
    # Stream reader (runs in daemon thread)
    # ------------------------------------------------------------------ #

    def _read_stream(self, stream, level: int) -> None:  # type: ignore[type-arg]
        """
        Read lines from *stream* until EOF, logging each with the process tag.
        """
        try:
            for raw_line in stream:
                line: str = raw_line.rstrip("\n\r")
                if line:
                    log.log(level, "[%s] %s", self.tag, line)
        except ValueError:
            # Stream closed while reading — expected during shutdown
            pass


# ---------------------------------------------------------------------------
# Server orchestrator
# ---------------------------------------------------------------------------


class ServerOrchestrator:
    """
    Launches, monitors, and tears down the API + UI child processes.
    """

    def __init__(self) -> None:
        self._shutting_down: bool = False
        self._processes: List[ManagedProcess] = []

    # ------------------------------------------------------------------ #
    # Port preflight checks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        """Return True when a TCP listener already exists on host:port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, int(port))) == 0

    def _preflight_ports(self) -> bool:
        """
        Validate that required ports are free before starting child processes.

        Returns
        -------
        True when both API and UI ports are available, False otherwise.
        """
        ok = True

        if self._is_port_in_use(API_PORT):
            log.critical(
                "API port %d is already in use. "
                "Stop the existing process on that port before running the process manager.",
                API_PORT,
            )
            ok = False

        if self._is_port_in_use(UI_PORT):
            log.critical(
                "UI port %d is already in use. "
                "Stop the existing process on that port before running the process manager.",
                UI_PORT,
            )
            ok = False

        return ok

    # ------------------------------------------------------------------ #
    # Build child commands
    # ------------------------------------------------------------------ #

    def _build_commands(self) -> List[ManagedProcess]:
        """
        Construct the exact CLI commands for Uvicorn and Streamlit using the
        same Python interpreter that is running this script.
        """
        python: str = sys.executable

        api = ManagedProcess(
            tag="API",
            cmd=[
                python, "-m", "uvicorn",
                "api.run_api:app",
                "--host", API_HOST,
                "--port", str(API_PORT),
                "--log-level", "info",
                "--no-access-log",
            ],
        )

        ui = ManagedProcess(
            tag="UI",
            cmd=[
                python, "-m", "streamlit", "run",
                str(ROOT_DIR / "frontend" / "app_enhanced.py"),
                "--server.port", str(UI_PORT),
                "--server.headless", "true",
                "--logger.level", "warning",
            ],
        )

        return [api, ui]

    # ------------------------------------------------------------------ #
    # Signal handlers
    # ------------------------------------------------------------------ #

    def _install_signal_handlers(self) -> None:
        """Register Ctrl+C and SIGTERM handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        sig_name: str = signal.Signals(signum).name
        log.info("Received %s — initiating graceful shutdown", sig_name)
        self._shutting_down = True

    # ------------------------------------------------------------------ #
    # Graceful teardown
    # ------------------------------------------------------------------ #

    def _teardown(self) -> None:
        """
        Phase 1: TERMINATE all children.
        Phase 2: Wait ``SHUTDOWN_GRACE_SECONDS``.
        Phase 3: KILL any survivors.
        """
        log.info("--- Teardown: Phase 1 (TERMINATE) ---")
        for mp in self._processes:
            mp.terminate()

        log.info(
            "--- Teardown: Waiting %.1f s for clean exit ---",
            SHUTDOWN_GRACE_SECONDS,
        )
        deadline: float = time.monotonic() + SHUTDOWN_GRACE_SECONDS
        while time.monotonic() < deadline:
            if all(mp.poll() is not None for mp in self._processes):
                break
            time.sleep(0.2)

        # Phase 3: force-kill stragglers
        for mp in self._processes:
            if mp.poll() is None:
                log.warning("[%s] Still alive after grace period — sending KILL", mp.tag)
                mp.kill()
                mp.wait(timeout=3.0)

        # Log final exit codes
        for mp in self._processes:
            code: Optional[int] = mp.poll()
            log.info("[%s] Exited with code %s", mp.tag, code)

    # ------------------------------------------------------------------ #
    # Watchdog loop
    # ------------------------------------------------------------------ #

    def _watchdog(self) -> None:
        """
        Poll child processes until one exits unexpectedly or a shutdown
        signal is received.

        If a child crashes (non-zero exit), the surviving sibling is
        terminated to prevent zombie/orphan states.
        """
        while not self._shutting_down:
            for mp in self._processes:
                code: Optional[int] = mp.poll()
                if code is None:
                    continue  # still running

                if code == 0:
                    log.info("[%s] Exited cleanly (code 0)", mp.tag)
                else:
                    log.critical(
                        "[%s] CRASHED with exit code %d", mp.tag, code,
                    )

                # Whether clean or crash — trigger full shutdown
                log.info(
                    "[%s] Process ended — shutting down remaining services",
                    mp.tag,
                )
                self._shutting_down = True
                return

            time.sleep(WATCHDOG_POLL_INTERVAL)

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        """
        Launch both services, run the watchdog, and return an exit code.

        Returns
        -------
        0 if both processes exited cleanly, 1 otherwise.
        """
        self._install_signal_handlers()
        self._processes = self._build_commands()

        log.info("=" * 72)
        log.info("APEX Process Manager starting")
        log.info("  API : http://localhost:%d", API_PORT)
        log.info("  UI  : http://localhost:%d", UI_PORT)
        log.info("  Logs: %s", LOG_FILE)
        log.info("  PID : %d (manager)", os.getpid())
        log.info("=" * 72)

        if not self._preflight_ports():
            log.info("Preflight failed; not starting child processes.")
            return 1

        # Start children
        try:
            for mp in self._processes:
                mp.start()
        except Exception as exc:
            log.critical("Failed to start child process: %s", exc, exc_info=True)
            self._teardown()
            return 1

        # Watchdog blocks until shutdown is triggered
        try:
            self._watchdog()
        except Exception as exc:
            log.critical("Watchdog error: %s", exc, exc_info=True)

        # Graceful teardown
        self._teardown()

        # Determine exit code
        codes: List[Optional[int]] = [mp.poll() for mp in self._processes]
        success: bool = all(c == 0 for c in codes)

        log.info("=" * 72)
        log.info(
            "APEX Process Manager stopped  [%s]",
            "OK" if success else "FAILURE",
        )
        log.info("=" * 72)

        return 0 if success else 1


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    exit_code: int = ServerOrchestrator().run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
