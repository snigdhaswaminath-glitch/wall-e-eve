# WALL-E and EVE's Signal

An autonomous agent that explores a simulated grid world, gathers clues under a
limited battery and time budget, and works out where EVE is. It keeps a
probabilistic belief about EVE's location, updates it from every observation,
plans its next action from that belief, and writes a full action trace and a
final mission report.

## Repository layout

```
wall-e-eve/
├── README.md
├── requirements.txt          # pyyaml, numpy
├── configs/
│   ├── sample_mission.yaml   # small, easy map
│   └── hard_mission.yaml     # larger map, low battery, noisy signal
├── src/
│   ├── env.py                # simulated environment
│   ├── belief.py             # Bayesian belief over EVE's location
│   ├── agent.py              # planner / decision logic
│   ├── run.py                # runs a mission, writes trace + report
│   └── report.py             # final mission report
└── outputs/                  # sample traces and reports (see below)
```

## Setup and run

```
pip install -r requirements.txt
python src/run.py configs/sample_mission.yaml
python src/run.py configs/hard_mission.yaml --out outputs/hard
```

`run.py` prints one line per step and writes two files into the output folder
(`outputs` by default, or whatever you pass to `--out`):

- `trace.jsonl`: one JSON line per step
- `report.md`: the final mission report

Each module also has a small demo: `python src/env.py <config>`,
`python src/belief.py <config>`, `python src/agent.py <config>`.

Everything is deterministic for a given config: the `seed` and `clues.seed`
fix the obstacles, EVE's position and the clues (the agent never sees any of
them directly).

## Environment design (`src/env.py`)

**World.** A `width x height` grid. Coordinates are `(x, y)` with the origin at
`(0, 0)`, x growing east and y growing north. Some cells are obstacles
(a fraction `map.obstacles` of the grid); the start cell and recharge stations
are always free. The agent does not know where obstacles are and learns about
them only by bumping into them.

**Hidden state.** EVE's position and the clues. EVE is placed roughly along the
initial signal's bearing, at a distance implied by the initial signal strength
(with some random jitter). Clues are placed on free cells near EVE. Each clue
carries a noisy hint: the bearing and distance from the clue to EVE.

**Signal model.** The signal strength at a cell is `exp(-distance_to_EVE / signal_scale)`,
optionally plus Gaussian noise (`signal_noise`). Bearings are in degrees with
0 = east and 90 = north.

**Actions.**

| Action | Effect | Battery cost |
|---|---|---|
| `{"type": "move", "dir": "N/S/E/W"}` | Move one cell. Walls and obstacles block the move (the cost is still paid). | `costs.move` |
| `{"type": "scan", "radius": r}` | Reveals clues within `r` cells and detects EVE if she is within `r`. `r` is capped at `max_scan_radius`. | `costs.scan` |
| `{"type": "recharge"}` | Refills the battery, only on a recharge station. Takes `costs.recharge_time` time units (default 5). | `costs.recharge` (after refilling) |
| `{"type": "abort"}` | Ends the mission (turning back). | none |

Every action also costs 1 time unit.

**Observation** returned by every `step(action)`: `t`, `pos`, `battery`,
`time_left`, `signal` (reading at the current cell), `new_clues` (clues revealed
by this action), `eve_found`, `done`, `status`, and a text `message`.

**Episode ends** when a scan finds EVE (`found_eve`), time runs out
(`out_of_time`), the battery is too low to move while away from a station
(`stranded`), or the agent aborts (`aborted`).

**Information boundary.** `env.public_info()` is everything the agent may know
in advance (map size, start, stations, costs, budgets, the initial signal, the
signal scale). `env.ground_truth()` exposes the hidden state and is used only
for scoring in the final report.

## Config format

YAML. Required fields:

| Field | Meaning |
|---|---|
| `seed` | Seed for obstacles and EVE's placement |
| `map.width`, `map.height` | Grid size |
| `map.obstacles` | Fraction of cells that are obstacles (0 to 1) |
| `start` | `[x, y]` starting cell |
| `initial_signal.bearing_deg` | Direction EVE's signal first came from |
| `initial_signal.strength` | Signal strength (0 to 1) at the start; implies a rough distance |
| `initial_signal.timestamp` | Recorded for completeness; the simulation does not use it (EVE does not move) |
| `budget.battery`, `budget.time` | Battery capacity and time limit |
| `costs.move`, `costs.scan`, `costs.recharge` | Battery cost of each action (`recharge` is the cost paid after refilling; 0 means a full battery) |
| `recharge_stations` | List of `[x, y]` cells where the battery can be refilled |
| `clues.count`, `clues.seed` | Number of clues and the seed that places them |

Optional fields (defaults in brackets):

| Field | Meaning |
|---|---|
| `signal_scale` [15] | How quickly the signal fades with distance |
| `signal_noise` [0] | Standard deviation of Gaussian noise on readings |
| `max_scan_radius` [4] | Largest allowed scan radius |
| `clue_spread` [8] | Clues are placed within this distance of EVE |
| `costs.recharge_time` [5] | Time units a recharge takes |

## Belief model (`src/belief.py`)

The belief is a grid of probabilities, one per cell ("EVE is here"), always
normalized to 1. It is updated with Bayes' rule (posterior is proportional to
prior times likelihood):

- **Prior:** a blob along the initial bearing, at the distance implied by the initial strength.
- **Signal reading:** every cell predicts a reading from the current position; cells whose prediction matches the real reading gain weight. Readings from different positions give different rings, and the overlap pins EVE down.
- **Clue:** boosts cells matching the clue's hinted bearing and distance from the clue's position.
- **Empty scan:** rules out every cell inside the scanned radius.
- **Successful scan:** restricts the belief to the scanned radius.

Outputs used elsewhere: best guess, entropy (bits), confidence (probability mass
within 2 cells of the best guess), top cells, and a status label
(`unknown` / `region narrowed` / `likely located` / `confirmed`). The model's
tolerance for signal noise is set from `signal_noise` in `run.py`.

## Agent (`src/agent.py`)

Each step the agent:

1. **Learns from its last action:** remembers blocked moves and which areas it has scanned.
2. **Picks a strategy phase** from the belief's confidence: `explore` (below 0.3), `triangulate` (0.3 to 0.7), or `close_in` (above 0.7). The phase sets how much weight goes to finding EVE with a scan versus moving closer. Phase switches are logged as strategy changes.
3. **Scores candidate actions** (move in four directions, or scan at the widest radius) as `(information gain + progress) / battery cost`:
   - *Move:* expected bits learned from the signal reading at the new cell, plus a small bonus for reducing the expected distance to EVE.
   - *Scan:* chance of finding EVE in the radius, plus the information from a hit or miss, plus a small bonus for covering unscanned ground (possible clues).
4. **Filters for safety:** any action that would leave too little battery to reach the nearest recharge station (a margin of 5 on top of the walking distance) is discarded.
5. **Handles low battery:** if nothing safe remains, it walks to the nearest station and recharges; if it cannot reach one, or it is already on a station with nothing safe left to do, it aborts. It also tops up the battery whenever it is on a station below 70% charge.
6. **Explains itself:** every decision returns a short reasoning string, such as `[triangulate] conf 0.44, entropy 5.0 bits, battery 100; best: move N (score 1.39), next: move E (1.39)`.

## Outputs

**`trace.jsonl`:** one line per step (step 0 is the starting observation):

```
t, action, observation {pos, battery, time_left, signal, new_clues, eve_found, status, message},
belief {best_guess, top_cells, entropy_bits, confidence}, phase, reasoning, candidate_scores
```

The belief in a line is the belief *after* absorbing that step's observation.

**`report.md`:** outcome and resource use; evidence found (strongest signal, clues
with the step they were found, empty scans); inferred location, status and
confidence; strategy changes; a recommended next step; and an evaluation section
comparing the estimate with the hidden ground truth (the agent never sees it).

## Sample runs

| Config | Output folder | Notes |
|---|---|---|
| `configs/sample_mission.yaml` | `outputs/` | Small map; EVE found in a handful of steps |
| `configs/hard_mission.yaml` | `outputs/hard/` | 30x30 map, 15% obstacles, battery 50, noisy signal; needs a recharge and finds two clues |

## Limitations

- EVE is stationary, and there is a single agent.
- Obstacles are unknown to the agent, so its battery-safety estimate uses walking distance without detours; the margin absorbs most of the error, but a mission can still end in `aborted` when detours eat the battery.
- The scan cost is flat in this environment, so the agent always scans at the widest radius.
- Not implemented: visualization, decoy signals, multi-agent coordination, an LLM planner comparison, and a multi-seed generalization benchmark.
