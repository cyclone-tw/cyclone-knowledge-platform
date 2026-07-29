"""``python -m ckp`` -- run the service with the resolved config.

Config resolution happens here rather than at import time so that importing
``ckp.app`` in a test never depends on the ambient environment.
"""

from __future__ import annotations

import sys

from ckp.app import create_app
from ckp.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    del argv  # no options yet; configuration is the config stack's job
    try:
        config = load_config()
    except ConfigError as exc:
        # Fail closed and loudly: a misconfigured deployment must not start
        # and quietly serve defaults.
        print(f"ckp: configuration error: {exc}", file=sys.stderr)
        return 2

    import uvicorn

    uvicorn.run(
        create_app(config),
        host=config.server_host,
        port=config.server_port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
