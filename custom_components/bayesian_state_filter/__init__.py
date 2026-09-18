"""Bayesian State Filter custom component."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.reload import async_setup_reload_service

from .const import DOMAIN


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up Bayesian State Filter and its YAML reload service."""
    await async_setup_reload_service(hass, DOMAIN, [Platform.SENSOR])
    return True
