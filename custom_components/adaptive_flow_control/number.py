"""Number platform for Adaptive Flow Control integration."""
import asyncio
from datetime import timedelta
import logging
from typing import Any, cast

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, STATE_ON, UnitOfTemperature
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback,
    Coroutine,
    Mapping,
)
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from . import AdaptiveFlowControlConfigEntry, async_save_controller
from .common import get_device_info
from .const import (
    ATTR_AUSSEN_TEMP,
    ATTR_AUSSEN_TREND_1H,
    ATTR_LAST_UPDATE,
    ATTR_MODEL_MAE,
    ATTR_NEXT_UPDATE,
    ATTR_RAUM_ABWEICHUNG,
    ATTR_RAUM_IST,
    ATTR_RAUM_SOLL,
    ATTR_VORLAUF_IST,
    CONF_AUSSEN_TEMP_SENSOR,
    CONF_IS_HEATING_ENTITY,
    CONF_HYSTERESE,
    CONF_MAX_SCHRITTWEITE,
    CONF_RAUM_IST_SENSOR,
    CONF_RAUM_SOLL_SENSOR,
    CONF_THERMISCHE_LEISTUNG_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_VORLAUF_IST_SENSOR,
    CONF_VORLAUF_SOLL_ENTITY,
    DEFAULT_HYSTERESE,
    DEFAULT_MAX_SCHRITTWEITE,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    RAUM_SOLL_DEBOUNCE_SECONDS,
    RAUM_SOLL_TRIGGER_GEWICHT,
)
from .types import SensorValues, VorlaufSollAndFeatures
from .const import HYSTERESE_MARGIN  # noqa: E402

# pylint: disable=hass-logger-capital, hass-logger-period
# ruff: noqa: BLE001
# ruff: logging-redundant-exc-info

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: AdaptiveFlowControlConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Flow Control number."""
    config = config_entry.data

    number = FlowControlNumber(
        hass=hass,
        config=config,
        config_entry=config_entry,
    )

    async_add_entities([number], True)


def clamp_vorlauf_soll(
    vorlauf: float,
    sensor_values: SensorValues,
    hysterese: float,
    max_vorlauf: float,
) -> float:
    """Clamp the prediction value to the HPSU switching rule.

    The HPSU only starts when VorlaufSoll >= Tv + 3 K, and stops
    when Tv >= VorlaufSoll + 3 K. With heating demand (raum_ist < raum_soll)
    a prediction near the current Tv lands in the dead zone and
    the heat pump cycles or never starts — therefore we clamp to at least
    Tv + Hysteresis + Margin.
    """
    if sensor_values.raum_ist >= sensor_values.raum_soll:
        return vorlauf

    min_vorlauf = sensor_values.vorlauf_ist + hysterese + HYSTERESE_MARGIN
    if vorlauf < min_vorlauf:
        return min(min_vorlauf, max_vorlauf)
    return vorlauf


def limit_schrittweite(
    neu: float,
    previous: float | None,
    max_schrittweite: float,
) -> float:
    """Limit the change of the setpoint per update cycle.

    A wrong learning event must not let the setpoint jump by many Kelvin
    — otherwise the HPSU (3K switching rule) runs uninterrupted for hours.
    No limit when previous is None (first run) or
    max_schrittweite <= 0.
    """
    if previous is None or max_schrittweite <= 0:
        return neu

    if neu > previous + max_schrittweite:
        return previous + max_schrittweite
    if neu < previous - max_schrittweite:
        return previous - max_schrittweite
    return neu


class FlowControlNumber(NumberEntity, RestoreEntity):
    """Representation of Flow Control Number Entity."""

    _attr_mode = NumberMode.BOX
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_icon = "mdi:thermometer-auto"

    def __init__(
        self,
        hass: HomeAssistant,
        config: Mapping[str, Any],
        config_entry: AdaptiveFlowControlConfigEntry,
    ) -> None:
        """Initialize the number."""
        self.hass = hass
        self._config = config
        self._config_entry = config_entry

        self._attr_device_info = get_device_info(config_entry.entry_id)

        # Sensor Entities
        self._aussen_temp_sensor = config[CONF_AUSSEN_TEMP_SENSOR]
        self._raum_ist_sensor = config[CONF_RAUM_IST_SENSOR]
        self._raum_soll_sensor = config[CONF_RAUM_SOLL_SENSOR]
        self._vorlauf_ist_sensor = config[CONF_VORLAUF_IST_SENSOR]
        self._vorlauf_soll_entity = config[CONF_VORLAUF_SOLL_ENTITY]
        self._is_heating_entity = config.get(CONF_IS_HEATING_ENTITY)
        self._thermische_leistung_sensor = config.get(CONF_THERMISCHE_LEISTUNG_SENSOR)

        # Configuration
        self._update_interval_minutes = config.get(
            CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
        )
        self._hysterese = config.get(CONF_HYSTERESE, DEFAULT_HYSTERESE)
        self._max_schrittweite = config.get(
            CONF_MAX_SCHRITTWEITE, DEFAULT_MAX_SCHRITTWEITE
        )

        # State
        self._attr_native_value = None
        self._extra_attributes = {}
        self._last_update = None
        self._next_update = None
        self._available = False

        # Heating mode tracking
        self._heating_resumed_at = None  # Track when heating resumed
        self._last_prediction_time = None  # Track last successful prediction

        # Listener
        self._update_interval_listener = None
        self._state_change_listener = None
        self._raum_soll_debounce_unsub = None
        self._ha_stop_unsub = None
        self._heating_state_listener = None

        # Number properties
        self._attr_name = "Adaptive Flow Control Vorlauf Soll"
        self._attr_unique_id = f"{DOMAIN}_{config_entry.entry_id}_vorlauf_soll"
        # Get min/max from controller
        self._attr_native_min_value = config_entry.runtime_data.min_vorlauf
        self._attr_native_max_value = config_entry.runtime_data.max_vorlauf
        self._attr_native_step = 0.1

        self._tasks: set[asyncio.Task[None]] = set()

        # Serializes update cycles: overlapping runs (interval timer,
        # sensor event, first start) would mutate the controller in
        # parallel (river model, erfahrungs_liste, HistoryBuffer) and
        # produce corrupt data during the pickle save.
        self._update_lock = asyncio.Lock()

    @property
    def _controller(self):
        """Get FlowController from runtime_data."""
        return self._config_entry.runtime_data

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        _LOGGER.info("async_added_to_hass()")

        await super().async_added_to_hass()

        # Restore previous state
        last_state = await self.async_get_last_state()
        if last_state is not None and last_state.state not in (
            "unavailable",
            "unknown",
        ):
            try:
                self._attr_native_value = float(last_state.state)
                _LOGGER.info("Restored state: %s", self._attr_native_value)
            except (ValueError, TypeError):
                pass

        # Setup periodic update
        update_interval = timedelta(minutes=self._update_interval_minutes)
        self._update_interval_listener = async_track_time_interval(
            self.hass,
            self._async_update_vorlauf_soll,
            update_interval,
        )

        # Setup state change listener
        sensor_list = [
            self._aussen_temp_sensor,
            self._raum_ist_sensor,
            self._raum_soll_sensor,
            self._vorlauf_ist_sensor,
        ]
        if self._thermische_leistung_sensor:
            sensor_list.append(self._thermische_leistung_sensor)

        self._state_change_listener = async_track_state_change_event(
            self.hass,
            sensor_list,
            self._async_sensor_state_changed,
        )

        # Shutdown-Save: async_will_remove_from_hass does NOT fire
        # on HA shutdown (HA does not unload entities when stopping). The STOP event
        # fires early enough during shutdown for the executor save to still
        # get going — unlearned experiences thus survive every restart.
        self._ha_stop_unsub = self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_on_ha_stop
        )

        # Setup heating state change listener if configured
        if self._is_heating_entity:
            self._heating_state_listener = async_track_state_change_event(
                self.hass,
                [self._is_heating_entity],
                self._async_heating_state_changed,
            )

        # Immediate first update after 5 seconds
        self._create_task(self._delayed_first_update())

    async def async_set_native_value(self, value: float) -> None:
        """Set new value (manual override)."""

        _LOGGER.info("async_set_native_value(%.2f)", value)

        self._attr_native_value = value

        # Send directly on manual change too (independent of the switch)
        await self._async_set_vorlauf_soll(value)
        self.async_write_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        _LOGGER.info("async_will_remove_from_hass())")

        # Write a snapshot before the entity is gone: learned experiences
        # (including unlearned ones from the last cycle) must not be lost
        # on shutdown/restart.
        if self._controller is not None and self._config_entry is not None:
            try:
                await async_save_controller(self.hass, self._config_entry, self._controller)
            except Exception:  # noqa: BLE001
                _LOGGER.warning(
                    "Model snapshot not saved on removal", exc_info=True
                )

        if self._update_interval_listener:
            self._update_interval_listener()
        if self._raum_soll_debounce_unsub:
            self._raum_soll_debounce_unsub()
            self._raum_soll_debounce_unsub = None
        if self._ha_stop_unsub:
            self._ha_stop_unsub()
            self._ha_stop_unsub = None
        if self._state_change_listener:
            self._state_change_listener()
        if self._heating_state_listener:
            self._heating_state_listener()

        for task in list(self._tasks):
            task.cancel()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""

        _LOGGER.debug("extra_state_attributes() %s", self._extra_attributes)

        return self._extra_attributes

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    # Private methods

    async def _async_on_ha_stop(self, _event) -> None:
        """Saves the model snapshot before HA shuts down."""

        _LOGGER.info("HA stopping: saving model snapshot")
        try:
            await async_save_controller(self.hass, self._config_entry, self._controller)
        except Exception:
            _LOGGER.warning(
                "Model snapshot not saved on HA stop", exc_info=True
            )

    async def _delayed_first_update(self):
        """First update after startup with delay."""
        await asyncio.sleep(5)
        await self._async_update_vorlauf_soll()

    @callback
    def _async_sensor_state_changed(self, event) -> None:
        """Handle sensor state changes for learning."""

        _LOGGER.debug("_async_sensor_state_changed()")

        # Ignore unavailable/unknown states
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in ("unavailable", "unknown"):
            return

        # If the Number is still unavailable, check whether all sensors are now available
        if not self._available:
            self._create_task(self._check_and_update_if_ready())
            return

        # raum_soll change: immediate (debounced) update so that the
        # heat pump does not keep regulating to the old setting until the
        # next hourly appointment. Only on a real value change, not on an attribute refresh.
        if (
            new_state.entity_id == self._raum_soll_sensor
            and event.data.get("old_state") is not None
            and event.data["old_state"].state != new_state.state
        ):
            self._schedule_raum_soll_update()

    @callback
    def _schedule_raum_soll_update(self) -> None:
        """Schedule a debounced update after a raum_soll change."""
        if self._raum_soll_debounce_unsub:
            self._raum_soll_debounce_unsub()
            self._raum_soll_debounce_unsub = None

        async def _debounced_update(_now) -> None:
            self._raum_soll_debounce_unsub = None
            _LOGGER.info(
                "raum_soll change: immediate update (debounced %ds)",
                RAUM_SOLL_DEBOUNCE_SECONDS,
            )
            # The experience of this immediate cycle learns damped: a setpoint
            # change contains the same information as a single
            # adjustment; multiple cycles must not amplify the model
            # N times over.
            await self._async_update_vorlauf_soll(durch_sollweechsel=True)

        self._raum_soll_debounce_unsub = async_call_later(
            self.hass, RAUM_SOLL_DEBOUNCE_SECONDS, _debounced_update
        )

    async def _check_and_update_if_ready(self):
        """Check if all sensors are ready and trigger update."""

        _LOGGER.debug("_check_and_update_if_ready()")

        sensor_values = await self._async_get_sensor_values()
        if sensor_values is not None:
            await self._async_update_vorlauf_soll()

    async def _is_heating_mode(self) -> bool:
        """Check if heat pump is in heating mode.

        Returns:
            True if in heating mode or sensor not configured, False otherwise
        """
        if not self._is_heating_entity:
            _LOGGER.warning("Entity is_heating_entity not configured!")
            return True  # No sensor configured = always learn

        is_heating_state = self.hass.states.get(self._is_heating_entity)
        if is_heating_state is None:
            _LOGGER.warning(
                "is_heating sensor %s not found, assuming heating disabled",
                self._is_heating_entity,
            )
            return False

        if is_heating_state.state in ("unavailable", "unknown"):
            _LOGGER.debug(
                "Is_heating entity %s is %s, assuming heating disabled",
                self._is_heating_entity,
                is_heating_state.state,
            )
            return False

        is_heating = is_heating_state.state == STATE_ON

        _LOGGER.info("_is_heating_mode: %s, %d", is_heating_state.state, is_heating)

        return is_heating

    async def _async_get_sensor_values(self) -> SensorValues | None:
        """Get current values from all sensors."""

        _LOGGER.info("_async_get_sensor_values()")

        try:
            aussen_temp_state_or_none = self.hass.states.get(self._aussen_temp_sensor)
            raum_ist_state_or_none = self.hass.states.get(self._raum_ist_sensor)
            raum_soll_state_or_none = self.hass.states.get(self._raum_soll_sensor)
            vorlauf_ist_state_or_none = self.hass.states.get(self._vorlauf_ist_sensor)

            _LOGGER.info(
                "Reading sensors: aussen=%s, raum_ist=%s, raum_soll=%s, vorlauf_ist=%s",
                self._format_sensor_as_float(aussen_temp_state_or_none),
                self._format_sensor_as_float(raum_ist_state_or_none),
                self._format_sensor_as_float(raum_soll_state_or_none),
                self._format_sensor_as_float(vorlauf_ist_state_or_none),
            )

            # Check for unavailable/unknown
            sensors = [
                (self._aussen_temp_sensor, aussen_temp_state_or_none),
                (self._raum_ist_sensor, raum_ist_state_or_none),
                (self._raum_soll_sensor, raum_soll_state_or_none),
                (self._vorlauf_ist_sensor, vorlauf_ist_state_or_none),
            ]

            for entity_id, state_obj in sensors:
                if state_obj is None:
                    _LOGGER.warning("Sensor %s state is None", entity_id)
                    return None
                if state_obj.state in ("unavailable", "unknown"):
                    _LOGGER.warning("Sensor %s is %s", entity_id, state_obj.state)
                    return None

            # Extract temperature values

            aussen_temp = float(cast(State, aussen_temp_state_or_none).state)
            raum_ist = float(cast(State, raum_ist_state_or_none).state)
            vorlauf_ist = float(cast(State, vorlauf_ist_state_or_none).state)
            raum_soll_state = cast(State, raum_soll_state_or_none)

            # raum_soll may come from a climate entity or an input number
            if raum_soll_state.domain == "climate":
                raum_soll = float(raum_soll_state.attributes.get("temperature", 21.0))
            else:
                raum_soll = cast(State, raum_soll_state)
                raum_soll_value = float(raum_soll.state)
                raum_soll = float(raum_soll_value)

            # Thermal power (optional)
            thermische_leistung = None
            if self._thermische_leistung_sensor:
                thermische_leistung_state = self.hass.states.get(self._thermische_leistung_sensor)
                if thermische_leistung_state and thermische_leistung_state.state not in ("unavailable", "unknown"):
                    try:
                        thermische_leistung = float(thermische_leistung_state.state)
                    except (ValueError, TypeError):
                        _LOGGER.warning("Could not parse thermische_leistung sensor value")

        except (ValueError, TypeError) as e:
            _LOGGER.error("Error reading sensor values: %s", e)
            return None

        else:
            return SensorValues(
                aussen_temp=aussen_temp,
                raum_ist=raum_ist,
                raum_soll=raum_soll,
                vorlauf_ist=vorlauf_ist,
                thermische_leistung=thermische_leistung,
            )

    async def _async_update_vorlauf_soll(
        self, now=None, durch_sollweechsel: bool = False
    ) -> None:
        """Update the Vorlauf-Soll value."""

        # Skip parallel calls: a running update cycle
        # must not be started twice by the timer/sensor event.
        if self._update_lock.locked():
            _LOGGER.debug("Update already in progress - skipping")
            return
        async with self._update_lock:
            await self._async_update_vorlauf_soll_locked(now, durch_sollweechsel)

    async def _async_update_vorlauf_soll_locked(
        self, now=None, durch_sollweechsel: bool = False
    ) -> None:
        """Update the Vorlauf-Soll value (serialized by _update_lock)."""

        is_heating = await self._is_heating_mode()

        _LOGGER.info("_async_update_vorlauf_soll() controller.enabled: %d, is_heating: %d", self._controller.enabled, is_heating)

        try:
            if not self._controller.enabled:
                _LOGGER.info("Controller disabled - skipping update")
                return

            if not is_heating:
                _LOGGER.info("Not in heating mode - skipping update")
                return

            if self._heating_resumed_at is not None:
                time_since_resume = (dt_util.now() - self._heating_resumed_at).total_seconds() / 60
                if time_since_resume < 10.0:
                    _LOGGER.info(
                        "In stabilization (%.1f/10.0 min) - skipping update",
                        time_since_resume
                    )
                    return

            # Normal operation - proceed with prediction
            sensor_values = await self._async_get_sensor_values()
            if sensor_values is None:
                _LOGGER.warning("sensor_values is None, setting unavailable")
                self._available = False
                self.async_write_ha_state()
                return

            # Calculate the new Vorlauf-Soll. Experiences from raum_soll-
            # triggered immediate cycles learn damped (factor < 1),
            # hourly cycles with full weight.
            gewicht_faktor = (
                RAUM_SOLL_TRIGGER_GEWICHT if durch_sollweechsel else 1.0
            )
            vorlauf_soll_and_features: VorlaufSollAndFeatures = await self.hass.async_add_executor_job(
                self._controller.calculate_vorlauf_soll,
                sensor_values,
                gewicht_faktor,
            )

            # 1. Step limit: only tames the MODEL value (protection against
            # model jumps, see limit_schrittweite). Runs BEFORE the
            # dead-zone clamp so that the deterministic heat pump start rule
            # below is not throttled — otherwise the heat pump would
            # only start hours later with heating demand.
            previous_value = (
                float(self._attr_native_value)
                if self._attr_native_value is not None
                else None
            )
            vorlauf_soll_effektiv = limit_schrittweite(
                vorlauf_soll_and_features.vorlauf, previous_value, self._max_schrittweite
            )
            if vorlauf_soll_effektiv != vorlauf_soll_and_features.vorlauf:
                _LOGGER.warning(
                    "Vorlauf-Soll jump limited: %.1f°C -> %.1f°C (max %.1fK per update)",
                    vorlauf_soll_and_features.vorlauf,
                    vorlauf_soll_effektiv,
                    self._max_schrittweite,
                )

            # 2. Dead-zone clamp: see clamp_vorlauf_soll(). A
            # prediction in the dead zone (< Tv+Hysteresis) is raised to
            # the heat pump start threshold with heating demand — unthrottled,
            # so the heat pump can start immediately.
            vorlauf_soll_geliefert = vorlauf_soll_effektiv
            vorlauf_soll_effektiv = clamp_vorlauf_soll(
                vorlauf_soll_effektiv,
                sensor_values,
                self._hysterese,
                self._controller.max_vorlauf,
            )
            if vorlauf_soll_effektiv != vorlauf_soll_geliefert:
                _LOGGER.info(
                    "Heating demand, prediction %.1f°C in dead zone (< Tv+%.1fK): clamped to %.1f°C",
                    vorlauf_soll_geliefert,
                    self._hysterese,
                    vorlauf_soll_effektiv,
                )

            # Reward attribution: The anchor of the freshly created experience
            # must match the value that is really sent to the heat pump
            # (after clamp and step limit).
            self._controller.set_delivered_vorlauf(vorlauf_soll_effektiv)

            # Track successful prediction time
            self._last_prediction_time = dt_util.now()

            # Update state
            self._attr_native_value = round(vorlauf_soll_effektiv, 1)
            self._last_update = dt_util.now()
            self._next_update = self._last_update + timedelta(
                minutes=self._update_interval_minutes
            )
            self._available = True

            # Update attributes
            model_statistics = self._controller.get_model_statistics()

            # Check is_heating (pump-agnostic gate, no HPSU operating mode sensor)
            current_is_heating = None
            if self._is_heating_entity:
                is_heating_state = self.hass.states.get(self._is_heating_entity)
                if is_heating_state and is_heating_state.state not in (
                    "unavailable",
                    "unknown",
                ):
                    current_is_heating = is_heating_state.state

            self._extra_attributes = {
                ATTR_AUSSEN_TEMP: round(sensor_values.aussen_temp, 1),
                ATTR_RAUM_IST: round(sensor_values.raum_ist, 1),
                ATTR_RAUM_SOLL: round(sensor_values.raum_soll, 1),
                ATTR_VORLAUF_IST: round(sensor_values.vorlauf_ist, 1),
                ATTR_RAUM_ABWEICHUNG: round(vorlauf_soll_and_features.features.raum_abweichung, 2),
                ATTR_AUSSEN_TREND_1H: round(vorlauf_soll_and_features.features.aussen_trend_1h, 3),
                ATTR_MODEL_MAE: round(model_statistics.mae, 2),
                ATTR_LAST_UPDATE: self._last_update.isoformat(),
                ATTR_NEXT_UPDATE: self._next_update.isoformat(),
                "history_size": model_statistics.history_size,
                "is_heating": current_is_heating,
                # Cumulative counter from the controller (do not read the
                # attribute value of our own — it is outdated when written)
                "erfahrungen_gelernt": model_statistics.erfahrungen_gelernt
            }

            # Check whether the switch is active, then send to the heat pump
            # The switch has unique_id = {DOMAIN}_{entry_id}_aktiv
            switch_state = None
            switch_unique_id = f"{DOMAIN}_{self._config_entry.entry_id}_aktiv"

            # Find the switch via unique_id in the entity registry
            entity_registry = er.async_get(self.hass)

            for entity_id in entity_registry.entities:
                entry = entity_registry.async_get(entity_id)
                if entry and entry.unique_id == switch_unique_id:
                    switch_state = self.hass.states.get(entity_id)
                    break

            # Fallback: try directly via the constructed entity ID
            if not switch_state:
                switch_entity_id = f"switch.{DOMAIN}_aktiv"
                switch_state = self.hass.states.get(switch_entity_id)

            _LOGGER.debug(
                "Checking switch: unique_id=%s, entity_id=%s, state=%s",
                switch_unique_id,
                switch_state.entity_id if switch_state else "NOT_FOUND",
                switch_state.state if switch_state else "NOT_FOUND",
            )

            if switch_state and switch_state.state == "on":
                await self._async_set_vorlauf_soll(vorlauf_soll_effektiv)
                _LOGGER.info("Control enabled, sending Vorlauf-Soll to heat pump")
            else:
                _LOGGER.info("Control disabled by switch, not sending Vorlauf-Soll")

            self.async_write_ha_state()

            _LOGGER.info(
                "Vorlauf-Soll updated: %.1f°C (Outdoor: %.1f°C, Room: %.1f/%.1f°C)",
                vorlauf_soll_effektiv,
                sensor_values.aussen_temp,
                sensor_values.raum_ist,
                sensor_values.raum_soll,
            )

            # Save after calculation (feature learning happens internally in the controller)
            await async_save_controller(self.hass, self._config_entry, self._controller)

        except Exception:
            _LOGGER.exception("Error updating Vorlauf-Soll")
            self._available = False
            self.async_write_ha_state()

    @callback
    def _async_heating_state_changed(self, event) -> None:
        """Handle heating mode state changes."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if new_state is None or old_state is None:
            return

        new_is_heating = new_state.state == STATE_ON
        old_is_heating = old_state.state == STATE_ON

        # Only react to transition from off to on
        if new_is_heating and not old_is_heating:
            _LOGGER.info("Heating resumed")

            # Check if update_interval has passed since last prediction
            if self._last_prediction_time is not None:
                time_since_last = (dt_util.now() - self._last_prediction_time).total_seconds() / 60
                if time_since_last < self._update_interval_minutes:
                    wait_time = self._update_interval_minutes - time_since_last
                    _LOGGER.info(
                        "Less than update_interval since last prediction: waiting %.1f more minutes (%.1f/%.1f min since last)",
                        wait_time,
                        time_since_last,
                        self._update_interval_minutes
                    )
                    # Stabilization will start at next scheduled update
                    return

            # update_interval passed or first run - start stabilization
            self._heating_resumed_at = dt_util.now()
            _LOGGER.info("Starting 10min stabilization period")
        elif not new_is_heating and old_is_heating:
            _LOGGER.info("Heating disabled")
            self._heating_resumed_at = None

    async def _async_set_vorlauf_soll(self, value: float) -> None:
        """Set the Vorlauf-Soll on the target entity."""

        _LOGGER.info("_async_set_vorlauf_soll()")

        try:
            target_entity = self.hass.states.get(self._vorlauf_soll_entity)
            if target_entity is None:
                _LOGGER.error("Target entity %s not found", self._vorlauf_soll_entity)
                return

            domain = target_entity.domain
            service_data = {
                "entity_id": self._vorlauf_soll_entity,
                "value": round(value, 1),
            }

            if domain == "input_number":
                await self.hass.services.async_call(
                    "input_number",
                    "set_value",
                    service_data,
                )
            elif domain == "number":
                await self.hass.services.async_call(
                    "number",
                    "set_value",
                    service_data,
                )
            elif domain == "climate":
                service_data = {
                    "entity_id": self._vorlauf_soll_entity,
                    "temperature": value,
                }
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    service_data,
                )
            else:
                _LOGGER.warning("Unsupported target entity domain: %s", domain)

        except Exception as e:
            _LOGGER.error("Error setting Vorlauf-Soll: %s", e)

    def _format_sensor_as_float(self, state: State | None) -> str:
        """Converts an HA state into a float string (2 digits) or 'None'."""

        if state is None or state.state in ("unknown", "unavailable"):
            return "None"
        try:
            return f"{float(state.state):.2f}"
        except (ValueError, TypeError):
            return "Invalid"

    def _create_task(self, coro: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
