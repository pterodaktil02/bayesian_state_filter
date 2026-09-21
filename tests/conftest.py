"""Test bootstrap for core tests without installing Home Assistant."""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The mathematical core has no Home Assistant dependency, but importing the
# custom-component package executes __init__.py. Keep the dev test environment
# lightweight by providing only the symbols imported there.
if "homeassistant" not in sys.modules:
    ha = types.ModuleType("homeassistant")
    const = types.ModuleType("homeassistant.const")
    core = types.ModuleType("homeassistant.core")
    helpers = types.ModuleType("homeassistant.helpers")
    reload_mod = types.ModuleType("homeassistant.helpers.reload")

    class _Platform:
        SENSOR = "sensor"

    class _HomeAssistant:
        pass

    async def _async_setup_reload_service(*args, **kwargs):
        return None

    const.Platform = _Platform
    core.HomeAssistant = _HomeAssistant
    reload_mod.async_setup_reload_service = _async_setup_reload_service

    sys.modules["homeassistant"] = ha
    sys.modules["homeassistant.const"] = const
    sys.modules["homeassistant.core"] = core
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.reload"] = reload_mod
