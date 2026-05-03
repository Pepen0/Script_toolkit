#!/usr/bin/env python3
from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from urllib.parse import quote

import uvicorn


HOST = "127.0.0.1"
PORT = 8765


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _run_server() -> None:
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, log_level="warning")


def main() -> None:
    # If another instance is already running, reuse it.
    if not _port_in_use(HOST, PORT):
        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()

        # Wait briefly for server bootstrap.
        for _ in range(100):
            if _port_in_use(HOST, PORT):
                break
            if not server_thread.is_alive():
                break
            time.sleep(0.1)
        if not _port_in_use(HOST, PORT):
            html = (
                "<h2>Transcript Extractor failed to start</h2>"
                "<p>Close this window and reopen the app.</p>"
                "<p>If it still fails, rebuild the app bundle.</p>"
            )
            webbrowser.open("data:text/html," + quote(html))
            return

    webbrowser.open(f"http://{HOST}:{PORT}")

    # Keep process alive while server thread runs.
    while _port_in_use(HOST, PORT):
        time.sleep(1)


if __name__ == "__main__":
    # Avoid tokenizers parallelism warning on some systems.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
