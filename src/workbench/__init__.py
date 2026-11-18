"""Face Biometric Research Workbench Backend Adapter Package.

Exposes existing P1, P1.1, P2, and P3 research implementations via a thin, typed HTTP/JSON
interface without duplicating algorithms or modifying frozen baselines.
"""

from __future__ import annotations

DEFAULT_WORKBENCH_HOST = "127.0.0.1"
DEFAULT_WORKBENCH_PORT = 6173

# Browser-blocked "unsafe" ports in the 6000-6999 range (WHATWG / Chromium port blocklist)
BROWSER_RESTRICTED_6000_PORTS: frozenset[int] = frozenset({6000, 6566, 6665, 6666, 6667, 6668, 6669, 6697})
