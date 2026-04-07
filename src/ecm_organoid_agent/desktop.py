from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from streamlit.web import bootstrap

try:
    from .workspace import ensure_workspace, resolve_project_dir
except ImportError:  # pragma: no cover
    from ecm_organoid_agent.workspace import ensure_workspace, resolve_project_dir

APP_TITLE = "ECM Organoid Research Desk"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the ECM Organoid Research Desk as a desktop app.")
    parser.add_argument(
        "--project-dir",
        default=None,
        type=Path,
        help="Optional workspace directory. Defaults to source project dir in dev mode and ~/ECM-Organoid-Research-Desk in bundled mode.",
    )
    parser.add_argument(
        "--port",
        default=None,
        type=int,
        help="Optional fixed local port for the embedded Streamlit server.",
    )
    parser.add_argument(
        "--internal-run-streamlit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.internal_run_streamlit:
        workspace = ensure_workspace(args.project_dir)
        os.environ["ECM_ORGANOID_PROJECT_DIR"] = str(workspace)
        frontend_path = Path(__file__).with_name("frontend.py")
        start_streamlit_server(
            frontend_path=frontend_path,
            port=args.port or 8501,
            project_dir=workspace,
        )
        return
    launch_desktop(project_dir=args.project_dir, port=args.port)


def launch_desktop(*, project_dir: Path | None = None, port: int | None = None) -> None:
    import webview

    workspace = ensure_workspace(project_dir)
    server_port = port or find_free_port()

    os.environ["ECM_ORGANOID_PROJECT_DIR"] = str(workspace)
    backend_process = start_backend_process(workspace=workspace, port=server_port)

    server_url = f"http://127.0.0.1:{server_port}"
    try:
        wait_for_server(server_url, timeout_seconds=60)

        webview.create_window(
            APP_TITLE,
            server_url,
            width=1440,
            height=960,
            min_size=(1160, 760),
            text_select=True,
        )
        webview.start()
    finally:
        stop_backend_process(backend_process)


def start_streamlit_server(*, frontend_path: Path, port: int, project_dir: Path) -> None:
    from streamlit import config as st_config

    os.environ["ECM_ORGANOID_PROJECT_DIR"] = str(project_dir)
    st_config.set_option("server.headless", True)
    st_config.set_option("server.port", port)
    st_config.set_option("browser.gatherUsageStats", False)
    st_config.set_option("server.fileWatcherType", "none")
    bootstrap.run(
        str(frontend_path),
        False,
        [],
        {},
    )


def start_backend_process(*, workspace: Path, port: int) -> subprocess.Popen[bytes]:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "desktop_backend.log"
    log_file = open(log_path, "ab")
    env = os.environ.copy()
    env["ECM_ORGANOID_PROJECT_DIR"] = str(workspace)

    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--internal-run-streamlit",
            "--project-dir",
            str(workspace),
            "--port",
            str(port),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "ecm_organoid_agent.desktop",
            "--internal-run-streamlit",
            "--project-dir",
            str(workspace),
            "--port",
            str(port),
        ]

    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    setattr(process, "_ecm_log_file", log_file)
    return process


def stop_backend_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_file = getattr(process, "_ecm_log_file", None)
    if log_file is not None and not log_file.closed:
        log_file.close()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def wait_for_server(base_url: str, *, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url, timeout=2) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Embedded Streamlit server did not start in time: {last_error}")


if __name__ == "__main__":
    main()
