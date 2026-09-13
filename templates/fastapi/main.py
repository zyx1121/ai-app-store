"""Entry point the platform starts: `uv run python main.py`.

The port is read here rather than passed on argv, because `start` in aias.yaml
is an argv array and nothing expands `$PORT` inside one. The host is not read
from anywhere: an app binds loopback, and the platform stops one that does not.
"""

import os

import uvicorn

HOST = "127.0.0.1"

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=int(os.environ.get("PORT", "8000")))
