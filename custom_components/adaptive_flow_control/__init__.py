"""Adaptive Flow Control integration for Home Assistant."""

import logging
from pathlib import Path
import pickle

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_LEARNING_RATE,
    CONF_MAX_THERMISCHE_LEISTUNG,
    CONF_MAX_VORLAUF,
    CONF_MIN_THERMISCHE_LEISTUNG,
    CONF_MIN_VORLAUF,
    CONF_SANITY_BAND,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_THERMISCHE_LEISTUNG,
    DEFAULT_MAX_VORLAUF_LO,
    DEFAULT_MIN_THERMISCHE_LEISTUNG,
    DEFAULT_MIN_VORLAUF_HI,
    DEFAULT_SANITY_BAND,
)
from .flow_controller import PICKLE_VERSION, FlowController
from .model_store import (
    controller_to_snapshot,
    lade_snapshot,
    snapshot_to_controller,
    speichere_snapshot,
)

# pylint: disable=hass-logger-capital, hass-logger-period
# ruff: logging-redundant-exc-info

_LOGGER = logging.getLogger(__name__)

DOMAIN = "adaptive_flow_control"
PLATFORMS = ["number", "switch"]

# Custom ConfigEntry type for runtime_data
type AdaptiveFlowControlConfigEntry = ConfigEntry[FlowController]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Adaptive Flow Control component."""
    return True


async def async_setup_entry(
    hass: HomeAssistant,
    entry: AdaptiveFlowControlConfigEntry
) -> bool:
    """Set up Adaptive Flow Control from a config entry."""
    _LOGGER.info("async_setup_entry()")

    # Load or create FlowController
    controller = await _async_load_or_create_controller(hass, entry)

    # Store in runtime_data
    entry.runtime_data = controller

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Listen for options updates
    entry.async_on_unload(entry.add_update_listener(update_listener))

    return True


async def _async_load_or_create_controller(
    hass: HomeAssistant, entry: ConfigEntry
) -> FlowController:
    """Load controller from disk or create new one."""

    # Get configuration
    min_vorlauf = entry.data.get(CONF_MIN_VORLAUF, DEFAULT_MIN_VORLAUF_HI)
    max_vorlauf = entry.data.get(CONF_MAX_VORLAUF, DEFAULT_MAX_VORLAUF_LO)
    learning_rate = entry.data.get(CONF_LEARNING_RATE, DEFAULT_LEARNING_RATE)
    min_thermische_leistung = entry.data.get(CONF_MIN_THERMISCHE_LEISTUNG, DEFAULT_MIN_THERMISCHE_LEISTUNG)
    max_thermische_leistung = entry.data.get(CONF_MAX_THERMISCHE_LEISTUNG, DEFAULT_MAX_THERMISCHE_LEISTUNG)
    sanity_band = entry.data.get(CONF_SANITY_BAND, DEFAULT_SANITY_BAND)

    # Model file paths: JSON snapshot (new) and legacy pickle (migration)
    json_file_name = f"{DOMAIN}.model.json"
    json_path = Path(hass.config.path(json_file_name))
    pkl_file_name = f"{DOMAIN}.model.pkl"
    pkl_path = Path(hass.config.path(pkl_file_name))

    def _lade_pkl() -> FlowController | None:
        """Loads the legacy pickle and migrates it to JSON."""

        if not pkl_path.exists():
            return None

        try:
            with pkl_path.open("rb") as f:
                loaded = pickle.load(f)

            # Check pickle version for migration
            loaded_version = getattr(loaded, "pickle_version", 1)
            if loaded_version < PICKLE_VERSION:
                _LOGGER.warning(
                    "Legacy-Model version mismatch (loaded: %d, current: %d). "
                    "Starting fresh to ensure compatibility.",
                    loaded_version,
                    PICKLE_VERSION,
                )
                return None

            # Legacy patches for old models
            loaded.use_fallback = False
            loaded.min_predictions_for_model = 0

            _LOGGER.info("Legacy model loaded successfully (version %d)", loaded_version)

            # Migration: save as JSON snapshot
            try:
                speichere_snapshot(controller_to_snapshot(loaded), json_path)
                _LOGGER.info("Migrated legacy model to %s", json_path)
            except (KeyError, TypeError, ValueError) as mig_err:
                _LOGGER.warning(
                    "Legacy model loaded but not migrated to JSON: %s. "
                    "It will be re-migrated on the next save.",
                    mig_err,
                )
            return loaded
        except (pickle.UnpicklingError, EOFError, AttributeError) as err:
            _LOGGER.error(
                "Cannot load legacy model (incompatible format or missing attributes): %s. "
                "Starting fresh.",
                err,
            )
            return None

    def _load() -> FlowController | None:
        """Load model from JSON snapshot (fallback: legacy pickle)."""
        _LOGGER.info("Loading model from %s", json_path)

        snapshot = lade_snapshot(json_path)
        if snapshot is None:
            _LOGGER.info("No saved model found, trying legacy pickle")
            return _lade_pkl()

        try:
            controller = snapshot_to_controller(
                snapshot,
                min_vorlauf=min_vorlauf,
                max_vorlauf=max_vorlauf,
                learning_rate=learning_rate,
                min_thermische_leistung=min_thermische_leistung,
                max_thermische_leistung=max_thermische_leistung,
                sanity_band=sanity_band,
            )
            _LOGGER.info(
                "Model restored from JSON (schema %s, %d experiences, %d history)",
                snapshot.get("schema_version"),
                len(controller.erfahrungs_liste),
                len(controller.aussen_temp_history),
            )
            return controller
        except (KeyError, TypeError, ValueError) as err:
            _LOGGER.error(
                "Cannot restore model from snapshot: %s. Starting fresh.", err
            )
            return None

    # Load controller from disk
    loaded_controller = await hass.async_add_executor_job(_load)

    if loaded_controller:
        # Update configuration parameters from current config
        loaded_controller.update_config(
            min_vorlauf=min_vorlauf,
            max_vorlauf=max_vorlauf,
            learning_rate=learning_rate,
            min_thermische_leistung=min_thermische_leistung,
            max_thermische_leistung=max_thermische_leistung,
            sanity_band=sanity_band,
        )
    else:
        # Create new controller
        _LOGGER.info("Creating new FlowController")
        loaded_controller = FlowController(
            min_vorlauf=min_vorlauf,
            max_vorlauf=max_vorlauf,
            learning_rate=learning_rate,
            min_thermische_leistung=min_thermische_leistung,
            max_thermische_leistung=max_thermische_leistung,
            sanity_band=sanity_band,
        )
        loaded_controller.setup(iterations=1)

    return loaded_controller


async def async_save_controller(
    hass: HomeAssistant,
    entry: ConfigEntry,
    controller: FlowController
) -> None:
    """Save controller to disk."""

    json_file_name = f"{DOMAIN}.model.json"
    json_path = Path(hass.config.path(json_file_name))
    pkl_file_name = f"{DOMAIN}.model.pkl"
    pkl_path = Path(hass.config.path(pkl_file_name))

    def _save():
        """Save model to disk."""
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            _LOGGER.debug("Saving model to %s", json_path)
            speichere_snapshot(controller_to_snapshot(controller), json_path)

            # Remove the legacy pickle once the JSON snapshot exists
            if pkl_path.exists():
                pkl_path.unlink()
                _LOGGER.info("Removed legacy model file %s", pkl_path)
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("Error saving model: %s", err)

    await hass.async_add_executor_job(_save)


async def update_listener(
    hass: HomeAssistant, entry: AdaptiveFlowControlConfigEntry
) -> None:
    """Handle options update."""
    _LOGGER.info("update_listener()")
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(
    hass: HomeAssistant, entry: AdaptiveFlowControlConfigEntry
) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
