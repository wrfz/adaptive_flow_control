# Heatpump Flow Control

**Language / Sprache:** English | [Deutsch](README.de.md)

ML-based supply-water-setpoint control for air-source heat pumps as a Home Assistant custom integration.

The integration replaces (or supplements) the fixed heating curve with an online learning model that adapts the supply water setpoint to the actual heating situation. The computed value is exposed as a `number` entity and passed on to the heat pump via automation, pyscript or a bridge — independent of the controller type (CAN, Modbus, MQTT, …).

---

## How it works

### Overview: the control loop

```
Sensors                       Integration                        Heat pump
─────────                     ───────────                        ─────────
Outdoor temp ──┐
Room ACTUAL ───┤   Feature vector ──► river model ──► Prediction
Room TARGET ───┤                                        │
Flow ACTUAL ───┘                                        ▼
                                      Safety chains (clamp,
                                      step limit, sanity band)
                                                       │
                                                       ▼
                                      number.heatpump_flow_control_vorlauf_soll
                                                       │
                                                       ▼  (via automation/pyscript)
                                      Heat pump supply setpoint
                                                       │
                                                       ▼  (4h later)
                                      Room temperature as "reward" ──► Learning
```

1. **Prediction:** A feature vector is built from the current sensor data; the river model predicts the supply setpoint.
2. **Delivery:** The prediction runs through several safety chains and lands as the value on the number entity, which the heat pump uses as its target flow temperature.
3. **Learning (time-shifted):** Every prediction is stored as an *experience*. Four hours later it is evaluated: how did the room temperature control error develop? From that the model learns a correction on the old prediction. This way the model learns from the *consequences* of its own decisions (delayed reward) instead of just the current state.

### Entities

| Entity | Type | Purpose |
|---|---|---|
| `number.heatpump_flow_control_vorlauf_soll` | Number | The computed supply setpoint (can be overwritten manually) |
| `switch.heatpump_flow_control_aktiv` | Switch | Control on/off — off = nothing is written, the model does not learn, all values stay persisted |

### Update cycle

An update runs hourly by default (additional triggers: state changes of the sensors). Before every calculation the integration checks:

1. **Switch on?** — No → skip.
2. **Is the heat pump heating?** — The configured `is_heating` binary sensor (e.g. derived controller-agnostically from CAN/Modbus: operating mode "heating" and no hot-water/DHW activity) must be `on`. During hot water or setback → skip, nothing is written and nothing is learned.
3. **Stabilization:** Right after heating starts, 10 minutes are waited so the system can settle after a start (prevents learning on transition values).

While an update runs, a parallel incoming update is skipped (lock).

**Instant update on target-value change:** In addition to the hourly timer, the integration reacts to changes of the `raum_soll` sensor — debounced by 60 seconds so that slider dragging or schedule changes do not trigger a chain of cycles. After the debounce, a full control cycle runs with all gates and safety chains; the heat pump then does not have to wait up to an hour for the new setting. (`raum_ist` changes deliberately do not trigger — they change constantly and would flood learning with redundancy.)

### Feature vector

Per cycle the model sees:

| Feature | Meaning |
|---|---|
| `aussen_temp` | current outdoor temperature |
| `raum_abweichung` | `room_target − room_actual` (negative = room too warm) |
| `aussen_trend_1h` | outdoor temperature trend of the last hour (°C/h) |
| `thermische_leistung_relativ` | current thermal power normalized to [min, max] (0–1) |
| `temp_diff` | outdoor temp − room ACTUAL |
| `stunde_sin`, `stunde_cos` | time of day, cyclically encoded |
| `wochentag_sin`, `wochentag_cos` | weekday, cyclically encoded |

*Note: The operating mode is deliberately not a feature — the is_heating gate filters heating operation up front, and the operating mode is solved controller-agnostically via the helper.*

### Model

- **Pipeline:** `river.compose.Pipeline(StandardScaler → LinearRegression(Adam))` — true online learning: every learning event adjusts the coefficients immediately, there is no batch phase.
- **Pre-training:** Before real operation the model is trained on synthetic data generated from the heating curve (temperature range −20 to +20 °C, all room deviations −5 to +5 K). Synthetic data points only count with weight `0.1` so real operating data dominates after a few days.
- **Fallback heating curve:** Interpolated from two anchor points: `(15 °C → min_supply)` and `(−20 °C → max_supply)`, plus `2 K per Kelvin` room deviation. It serves as reference for the sanity band and as a fallback value.

### The safety chains (in execution order)

| Stage | What | Protection against |
|---|---|---|
| 1. **Sanity band** (`sanity_band`, default 8 K) | If the prediction misses the heating curve by more than ±8 K, it counts as a model error (e.g. extrapolation onto never-seen feature combinations) and is replaced by the heating curve (log: `Model liefert unrealistischen Wert … verwende Fallback-Heizkurve`). | Wild jumps of the model |
| 2. **Range limits** | Clamping to `[min_supply, max_supply]`. | physically nonsensical values |
| 3. **Step limit** (`max_schrittweite`, default 1 K) | Max. change of the **model value** per cycle (log: `Sollwert-Sprung begrenzt …`). Invisible in normal operation (hourly changes < 1 K); it only catches jumps and is a permanent guardrail, not a temporary patch. | a wrong learning event lets the heat pump run full blast for hours |
| 4. **Dead-zone clamp** (`hysterese`, default 3 K) | Many heat pump controllers (e.g. HPSU) only start when `SupplySetpoint ≥ flow_actual + 3 K`. With heating demand (`room_actual < room_target`) a prediction in this dead zone is clamped to at least `flow_actual + 3.5 K` (log: `Heizbedarf, Vorhersage … in toter Zone … geklemmt`). The clamp runs **after** the step limit and is therefore **not throttled** — the controller's deterministic start rule acts immediately while model jumps stay limited. | short cycling / no start / delayed start after setpoint change |

The step limit is **never switched off automatically**. If the model later learns reliably enough that faster adaptation is desired: increase the `max_schrittweite` option (e.g. 2 K) or set it to `0` (= off).

### Time-shifted learning (delayed reward)

During the update an **experience** is stored: timestamp, feature vector and the **actually delivered** setpoint (after clamps and step limit — not the raw prediction). Four hours later, at the next update:

1. Evaluation of the experience: How far is the room from the target (`deviation = room_target − room_actual_now`)?
   - Room too cold → the delivered flow temperature was too low → learning target is **raised**.
   - Room too warm → learning target is **lowered**.
2. The learning target value is `delivered_supply + deviation × 1.5` (damped correction, `KORREKTUR_FAKTOR`), clamped to `[min_supply, max_supply]`.
3. The weight of the learning step decreases with the age of the experience (24-h decay, `KORREKTUR_DECAY_HOURS`) and increases with the observed thermal power, max 5.0 (`MAX_WEIGHT`). If the correction factor is below 0.1, the experience is skipped instead of learning a self-confirmation.
4. Learned experiences older than 7 days are cleaned up; a cumulative counter (`erfahrungen_gelernt` attribute) is preserved.

### Persistence

- Snapshot: `/config/heatpump_flow_control.model.json` (model, experiences, counters) — saved at the end of every update cycle, when the integration is removed/reloaded, **and** when Home Assistant shuts down (STOP event), so unlearned experiences survive every restart.
- On startup the snapshot is loaded automatically (`Model restored from JSON …` in the log). If it does not exist, the model is set up fresh and pre-trained synthetically.
- Manual intervention: The setpoint can be set directly on the number entity at any time (e.g. during a holiday absence). The next regular cycle resumes control.

---

## Configuration

The integration is set up via *Settings → Devices & Services → Add Integration*. All parameters can later be changed via *Configure* (options flow) and take effect without a restart.

### Required (sensors)

| Parameter | Description |
|---|---|
| **Outdoor temp sensor** | `sensor.*` with outdoor temperature in °C |
| **Room actual sensor** | `sensor.*` with the current room temperature (reference room) |
| **Room target sensor** | `sensor.*`, `input_number.*` or `climate.*` that provides the desired setpoint |
| **Flow actual sensor** | `sensor.*` with the current flow temperature (Tv) |
| **Supply target entity** | The target: `number.*`, `input_number.*` or `climate.*` into which the computed setpoint is written (for CAN-/Modbus-coupled heat pumps typically a `number` that pyscript/bridge writes into the controller setpoint) |
| **is_heating entity** | `binary_sensor.*` — `on` = heat pump is heating (not hot water/DHW). Build controller-agnostically, e.g. from operating mode + DHW signal |

### Optional parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| **Thermal power sensor** | – | – | `sensor.*` with thermal power (kW). Improves the learning weighting; without a sensor the mean 0.5 is used |
| **min_supply** | 25 °C | 25–32 | Lower limit of the setpoint + heating curve anchor at 15 °C |
| **max_supply** | 55 °C | 35–55 | Upper limit of the setpoint + heating curve anchor at −20 °C |
| **min/max thermal power** | 0.6 / 6.0 kW | – | Scaling reference for `thermische_leistung_relativ` |
| **update_interval** | 60 min | 1–240 | Cycle time of the control |
| **learning_rate** | 0.01 | 0.001–0.1 | Adam learning rate of the model. Small = slow, stable; large = fast, nervous |
| **hysterese** | 3 K | 0–10 | Switching-rule distance of the heat pump controller for the dead-zone clamp (`flow_actual + hysterese + 0.5 K`) — adapt to your controller's start threshold |
| **max_schrittweite** | 1 K | 0–5 | Max. setpoint change per cycle. `0` = off |
| **sanity_band** | 8 K | 1–20 | Tolerance band around the heating curve. Do not choose too narrow — a trained model is legitimately allowed to deviate from the heating curve (observed in practice: ~5.5 K at low outdoor temperatures) |

### Number entity attributes (debug window)

| Attribute | Meaning |
|---|---|
| `aussen_temp`, `raum_ist`, `raum_soll`, `vorlauf_ist` | sensor values last used |
| `raum_abweichung`, `aussen_trend_1h` | current features |
| `is_heating` | state of the is_heating gate at the last update |
| `erfahrungen_gelernt` | cumulative counter of all learning events |
| `history_size` | length of the outdoor temperature trend buffer (not the learning history!) |
| `model_mae` | running model error |
| `last_update` / `next_update` | last/next scheduled control run |

---

## Examples

### Example 1: Normal winter day (all chains unremarkable)

State: outdoor 9.7 °C, room 23.0 °C, room target 22.5 °C, flow actual 30.5 °C, last setpoint 30.5 °C.

1. Heating curve fallback: `25 + (40−25)/35 × (15−9.7) + 2 × (22.5−23.0)` ≈ **26.3 °C** → sanity band [18.3; 34.3].
2. Model prediction: **31.9 °C** → inside the band → accepted.
3. Range limits [25; 40]: unchanged.
4. Dead-zone clamp: room is warm enough (23.0 ≥ 22.5) → **no** heating demand → no clamp needed.
5. Step limit: 31.9 > 30.5 + 1.0 → **limited to 31.5 °C** (log: `Sollwert-Sprung begrenzt: 31.9°C → 31.5°C (max 1.0K pro Update)`).
6. Delivered: **31.5 °C** — and exactly this value lands in the experience for later learning.

### Example 2: Dead-zone clamp prevents heat pump cycling and delayed start

State: room 21.8 °C, target 22.5 °C → heating demand. Flow actual 30.1 °C. Model predicts **28.0 °C**, last delivered value 30.5 °C.

- The heat pump only starts at `setpoint ≥ flow_actual + 3 K` = 33.1 K. A setpoint of 28.0 °C would sit in the dead zone: the pump never starts or cycles.
- 1. Step limit: 28.0 < 30.5 − 1 → limited to **29.5 °C** (model protection).
- 2. Dead-zone clamp: `min_setpoint = 30.1 + 3.0 + 0.5 = 33.6 °C` → delivered is **33.6 °C** (log: `Heizbedarf, Vorhersage 29.5°C in toter Zone (< Tv+3.5K): geklemmt auf 33.6°C`) — **not throttled**, the pump starts immediately.
- If the clamp ran (as it used to) before the step limit, it would have squeezed the setpoint to 31.5 °C and the pump would only have started after 2–3 further cycles.
- The experience stores 33.6 °C (the delivered value), not the 28.0.

### Example 3: Sanity band catches extrapolation

At night at −2 °C outdoor temperature the model delivers **39.2 °C**, although the heating curve says ~26 °C.

- Band ±8 K → [18; 34]. 39.2 is outside → model error.
- Delivered is the **heating curve (26 °C)** (log: `Model liefert unrealistischen Wert 39.2°C (Heizkurve: 26.0°C, Band ±8.0K), verwende Fallback-Heizkurve`).
- For learning the incident stays harmless: here too the experience carries the delivered value as anchor.

### Example 4: Time-shifted learning

4.1 h ago the following was delivered: 28.4 °C at room target 22.5 °C. Now: room 21.8 °C (too cold).

- Deviation: `22.5 − 21.8 = +0.7 K` → heated too little.
- Learning target: `28.4 + 0.7 × 1.5 = 29.5 °C`.
- The model learns the target 29.5 °C with the features from back then, weighted by age (24-h decay) and thermal power (log: `✓ Gelernt aus Erfahrung von vor 4.1h: Vorlauf 28.4°C → 29.5°C`).
- The `erfahrungen_gelernt` counter increases by 1.

### Example 5: Hot water aborts the cycle

12:48 — the heat pump switches to hot water (is_heating goes `off`).

- Next update 13:48: gate checks `is_heating` → `off` → **skip**: no new setpoint, no experience, no learning. The last delivered value stays until heating operation resumes.
- 10 minutes after returning to heating operation the first update is skipped (stabilization), afterwards the integration controls normally again.

### Example 6: Coupling to the heat pump

The integration only writes its own number entity. Handing the value to the heat pump controller is deliberately kept external and controller-specific — typical ways: pyscript, automation with `number.set_value`, Modbus/MQTT bridge. Example for a CAN-coupled HPSU via pyscript:

```yaml
# pyscript/heat_pump_set_vorlauf_soll_geregelt.py — simplified
@state_trigger("number.heatpump_flow_control_vorlauf_soll")
def copy_to_controller():
    setpoint = number.heatpump_flow_control_vorlauf_soll
    number.set_value("number.hpsu_can_vorlauf_soll_geregelt", str(setpoint))
```

Important: always make manual interventions on the **ML number**, not on the copy for the controller — otherwise the copy cycle overwrites the input within seconds.

---

## Logging / Troubleshooting

Enable DEBUG logging for the integration: *Settings → Devices & Services → Heatpump Flow Control → three dots → Enable debug logging* (or via developer tools). Flow of a cycle in the log:

```
_async_update_vorlauf_soll() controler.enabled: 1, is_heating: 1
_is_heating_mode: on, 1
_async_get_sensor_values() → Reading sensors: aussen=9.7, raum_ist=23.0, …
_erstelle_features()
Model-Prediction: 31.90°C (Außen: 9.7°C, Raum-Abw: -0.48°C, Vorlauf-Ist: 30.5°C)
Vorlauf-Soll berechnet: 31.5°C …
Erfahrung bewertet / ✓ Gelernt aus Erfahrung von vor …
Saving model to /config/heatpump_flow_control.model.json
```

Warnings to watch for:

| Log line | Meaning |
|---|---|
| `raum_soll-Änderung: sofortiges Update (entprellt 60s)` | Setpoint was changed — full control cycle runs 60 s after the last change |
| `Sollwert-Sprung begrenzt` | Step limit kicked in — watch whether it happens often in normal operation (then `max_schrittweite` would be too tight or the model nervous) |
| `Model liefert unrealistischen Wert` | Sanity band kicked in — frequent occurrence indicates missing/distorted training data |
| `Heizbedarf, Vorhersage … in toter Zone` | Dead-zone clamp kicked in — normal with small room deviations; raises to the heat pump's start threshold |
| `Not in heating mode - skipping update` | Gate: no heating operation (hot water/setback) — normal |

(The log lines are German because the integration logs in its native language.)

---

## Architecture files

| File | Contents |
|---|---|
| `custom_components/heatpump_flow_control/__init__.py` | Setup, restore/snapshot IO |
| `custom_components/heatpump_flow_control/flow_controller.py` | Control core: features, model, heating curve, learning, experiences |
| `custom_components/heatpump_flow_control/number.py` | Number/switch entities, update cycle, safety chains, attributes |
| `custom_components/heatpump_flow_control/model_store.py` | JSON persistence |
| `custom_components/heatpump_flow_control/config_flow.py` | Setup and options dialogs |
| `custom_components/heatpump_flow_control/const.py` | Constants and defaults |
| `tests/` | Unit tests (`pytest tests/`) |
