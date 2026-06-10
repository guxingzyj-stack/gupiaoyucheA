import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


APP_NAME = "GuangHuanStock"
PORT = int(os.environ.get("GUANGHUAN_STOCK_PORT", "8501"))


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    existing = os.environ.get("GUANGHUAN_STOCK_DATA_DIR")
    if existing:
        return Path(existing)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return _project_root() / "app"


def _python_exe(root: Path) -> Path:
    bundled = root / "python" / "python.exe"
    if bundled.exists():
        return bundled
    return Path(sys.executable)


def _start_streamlit(root: Path, data_dir: Path) -> None:
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "streamlit_startup.log"

    env = os.environ.copy()
    env["GUANGHUAN_STOCK_DATA_DIR"] = str(data_dir)
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        str(_python_exe(root)),
        "-m",
        "streamlit",
        "run",
        str(root / "app" / "app.py"),
        "--server.port",
        str(PORT),
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    with open(log_file, "a", encoding="utf-8") as log:
        log.write("\n\n=== launch ===\n")
        log.write(" ".join(cmd) + "\n")
        subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log,
            stderr=log,
            creationflags=creationflags,
        )


def main() -> int:
    root = _project_root()
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not _is_port_open(PORT):
        _start_streamlit(root, data_dir)
        for _ in range(40):
            if _is_port_open(PORT):
                break
            time.sleep(0.5)

    webbrowser.open(f"http://localhost:{PORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
