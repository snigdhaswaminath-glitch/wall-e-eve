"""env.py - simulated world for "WALL-E and EVE's Signal".

The environment loads a mission config, hides EVE and the clues, and returns
an observation after every action. The agent only ever sees observations;
it must NOT read env.eve or env.clues directly (use ground_truth() only for
evaluation and the final report).

Coordinates: (x, y), origin (0, 0), x grows east, y grows north.
Bearings: degrees, 0 = east, 90 = north (standard math convention).
"""
import math
import random

import numpy as np
import yaml

DIRS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        self.w = cfg["map"]["width"]
        self.h = cfg["map"]["height"]
        self.start = tuple(cfg["start"])
        self.stations = [tuple(s) for s in cfg["recharge_stations"]]
        self.costs = cfg["costs"]
        self.max_battery = cfg["budget"]["battery"]
        self.max_time = cfg["budget"]["time"]
        # Optional settings with defaults
        self.signal_scale = cfg.get("signal_scale", 15.0)  # signal = exp(-dist/scale)
        self.signal_noise = cfg.get("signal_noise", 0.0)   # std dev of reading noise
        self.max_scan_radius = cfg.get("max_scan_radius", 4)
        self.recharge_time = self.costs.get("recharge_time", 5)
        self.clue_spread = cfg.get("clue_spread", 8)       # clues lie within this of EVE
        self._build_world()
        self.reset()

    # ------------------------------------------------------------------ world
    def _build_world(self):
        rng = np.random.default_rng(self.cfg["seed"])
        self.noise_rng = np.random.default_rng(self.cfg["seed"] + 1)

        # Obstacles (never on the start cell or a recharge station)
        p = self.cfg["map"].get("obstacles", 0.0)
        self.blocked = rng.random((self.w, self.h)) < p
        for (x, y) in [self.start] + self.stations:
            self.blocked[x][y] = False

        # EVE: placed along the initial bearing at a distance implied by the
        # initial signal strength, with some jitter so it is not exact.
        sig = self.cfg["initial_signal"]
        strength = min(max(sig["strength"], 0.01), 0.999)
        dist = -self.signal_scale * math.log(strength) * rng.uniform(0.8, 1.2)
        bearing = math.radians(sig["bearing_deg"] + rng.normal(0, 15))
        tx = int(round(self.start[0] + dist * math.cos(bearing)))
        ty = int(round(self.start[1] + dist * math.sin(bearing)))
        tx = min(max(tx, 0), self.w - 1)
        ty = min(max(ty, 0), self.h - 1)
        self.eve = self._nearest_free(tx, ty, exclude={self.start})

        # Clues: random free cells near EVE, each with a noisy hint about EVE.
        crng = np.random.default_rng(self.cfg["clues"]["seed"])
        candidates = [
            (i, j)
            for i in range(self.w)
            for j in range(self.h)
            if not self.blocked[i][j]
            and (i, j) not in (self.start, self.eve)
            and math.hypot(i - self.eve[0], j - self.eve[1]) <= self.clue_spread
        ]
        n = min(self.cfg["clues"]["count"], len(candidates))
        picks = crng.choice(len(candidates), size=n, replace=False)
        self.clues = []
        for k, idx in enumerate(picks):
            cx, cy = candidates[idx]
            true_bearing = math.degrees(math.atan2(self.eve[1] - cy, self.eve[0] - cx))
            true_dist = math.hypot(self.eve[0] - cx, self.eve[1] - cy)
            self.clues.append({
                "id": k,
                "pos": (cx, cy),
                "hint": {
                    "bearing_to_eve": round((true_bearing + crng.normal(0, 10)) % 360, 1),
                    "distance_to_eve": round(max(0.0, true_dist + crng.normal(0, 1.5)), 1),
                },
            })

    def _nearest_free(self, x, y, exclude=()):
        best, best_d = None, float("inf")
        for i in range(self.w):
            for j in range(self.h):
                if self.blocked[i][j] or (i, j) in exclude:
                    continue
                d = math.hypot(i - x, j - y)
                if d < best_d:
                    best, best_d = (i, j), d
        return best

    # ------------------------------------------------------------ agent-facing
    def public_info(self):
        """Everything the agent is allowed to know before it starts."""
        return {
            "width": self.w,
            "height": self.h,
            "start": self.start,
            "recharge_stations": list(self.stations),
            "costs": dict(self.costs),
            "max_battery": self.max_battery,
            "max_time": self.max_time,
            "max_scan_radius": self.max_scan_radius,
            "signal_scale": self.signal_scale,
            "initial_signal": dict(self.cfg["initial_signal"]),
        }

    def reset(self):
        self.pos = self.start
        self.battery = self.max_battery
        self.time_left = self.max_time
        self.t = 0
        self.found_ids = set()
        self.eve_found = False
        self.done = False
        self.status = "running"
        return self._obs("mission start")

    def signal_at(self, pos):
        d = math.hypot(pos[0] - self.eve[0], pos[1] - self.eve[1])
        s = math.exp(-d / self.signal_scale)
        if self.signal_noise > 0:
            s += self.noise_rng.normal(0, self.signal_noise)
        return round(min(max(s, 0.0), 1.0), 4)

    def _obs(self, message, new_clues=None):
        return {
            "t": self.t,
            "pos": self.pos,
            "battery": self.battery,
            "time_left": self.time_left,
            "signal": self.signal_at(self.pos),
            "new_clues": new_clues or [],
            "eve_found": self.eve_found,
            "done": self.done,
            "status": self.status,
            "message": message,
        }

    def step(self, action):
        """action examples:
        {"type": "move", "dir": "N"}   dir in N/S/E/W
        {"type": "scan", "radius": 3}
        {"type": "recharge"}           only works on a recharge station
        {"type": "abort"}              turn back / end the mission
        """
        if self.done:
            raise RuntimeError("Episode is over; call reset().")
        self.t += 1
        self.time_left -= 1  # every action costs 1 time unit
        kind = action["type"]
        new_clues, msg = [], ""

        if kind == "abort":
            self.done, self.status = True, "aborted"
            return self._obs("agent aborted the mission")

        cost = self.costs.get(kind, 0)
        if kind != "recharge" and self.battery < cost:
            msg = "insufficient battery; action not performed"
        elif kind == "move":
            dx, dy = DIRS[action["dir"]]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            self.battery -= cost
            if 0 <= nx < self.w and 0 <= ny < self.h and not self.blocked[nx][ny]:
                self.pos = (nx, ny)
                msg = f"moved {action['dir']}"
            else:
                msg = f"blocked moving {action['dir']} (wall or obstacle)"
        elif kind == "scan":
            r = min(int(action.get("radius", 2)), self.max_scan_radius)
            self.battery -= cost
            for c in self.clues:
                if c["id"] not in self.found_ids and math.hypot(
                    c["pos"][0] - self.pos[0], c["pos"][1] - self.pos[1]
                ) <= r:
                    self.found_ids.add(c["id"])
                    new_clues.append(c)
            if math.hypot(self.eve[0] - self.pos[0], self.eve[1] - self.pos[1]) <= r:
                self.eve_found = True
            msg = f"scanned radius {r}: {len(new_clues)} new clue(s)"
        elif kind == "recharge":
            if self.pos in self.stations:
                self.battery = self.max_battery - cost
                self.time_left -= self.recharge_time - 1
                msg = "recharged at station"
            else:
                msg = "no recharge station here"
        else:
            msg = f"unknown action {kind!r}"

        # Termination checks
        if self.eve_found:
            self.done, self.status = True, "found_eve"
        elif self.time_left <= 0:
            self.done, self.status = True, "out_of_time"
        elif self.battery < self.costs["move"] and self.pos not in self.stations:
            self.done, self.status = True, "stranded"
        return self._obs(msg, new_clues)

    # ----------------------------------------------------- evaluation only
    def ground_truth(self):
        """For scoring and the final report. The agent must not call this."""
        return {"eve": self.eve, "clues": self.clues}


if __name__ == "__main__":
    # Smoke test: a random agent, just to check the environment runs.
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/sample_mission.yaml"
    env = Env(load_config(path))
    obs = env.reset()
    print("start:", obs)
    while not env.done:
        if random.random() < 0.2:
            action = {"type": "scan", "radius": 3}
        else:
            action = {"type": "move", "dir": random.choice(list(DIRS))}
        obs = env.step(action)
    print("end:", obs)
    print("truth:", env.ground_truth()["eve"])
