"""agent.py - the planner that decides WALL-E's next action.

Each step the agent:
  1. learns from the result of its last action (blocked moves, scanned area)
  2. picks a strategy phase from how confident the belief is
  3. lists candidate actions (move N/S/E/W, scan) and scores each as
        (information gained + progress toward EVE) / battery cost
  4. throws out any action that would leave too little battery to reach a
     recharge station, and recharges / returns / aborts when needed
  5. returns (action, reasoning string) for the trace log

It only uses env.public_info() and observations, never the hidden state.
"""
import math

import numpy as np

DIRS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


def h2(p):
    """Entropy (bits) of a yes/no event with probability p."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class Agent:
    # phase -> (weight on "find EVE with a scan", weight on "get closer to EVE")
    PHASES = {
        "explore": (2.0, 0.1),      # belief is vague: gather information
        "triangulate": (3.0, 0.25),  # belief is forming: balance both
        "close_in": (6.0, 0.5),      # belief is sharp: go there and confirm
    }

    def __init__(self, info, safety_margin=5, clue_bonus=0.5):
        self.w, self.h = info["width"], info["height"]
        self.start = tuple(info["start"])
        self.stations = [tuple(s) for s in info["recharge_stations"]] or [self.start]
        self.move_cost = info["costs"]["move"]
        self.scan_cost = info["costs"]["scan"]
        self.recharge_time = info["costs"].get("recharge_time", 5)
        self.max_battery = info["max_battery"]
        self.scan_radius = info["max_scan_radius"]  # flat scan cost, so always scan widest
        self.margin = safety_margin
        self.clue_bonus = clue_bonus

        self.X, self.Y = np.meshgrid(np.arange(self.w), np.arange(self.h), indexing="ij")
        self.visits = {}  # cell -> how many times WALL-E has stood there
        self.scanned = np.zeros((self.w, self.h), dtype=bool)
        self.blocked_moves = set()  # (cell, direction) pairs that hit a wall/obstacle
        self.last_action = None
        self.phase = None
        self.strategy_changes = []  # (t, old_phase, new_phase, confidence)
        self.last_scores = []       # [(description, score), ...] from the latest decision

    # ---------------------------------------------------------------- helpers
    def _dist(self, pos):
        return np.hypot(self.X - pos[0], self.Y - pos[1])

    def _home(self, pos):
        """(steps, station) for the nearest recharge station by walking distance."""
        return min(
            (abs(pos[0] - s[0]) + abs(pos[1] - s[1]), s) for s in self.stations
        )

    def _home_cost(self, pos):
        return self._home(pos)[0] * self.move_cost

    def _in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def _reading_info(self, belief, pos):
        """Bits we expect to learn from the signal reading at `pos`.

        Every hypothesis (cell) predicts a different reading. If the predicted
        readings are spread out, the reading will tell us a lot; if they all
        look alike, it will tell us little.
        """
        readings = np.exp(-self._dist(pos) / belief.scale)
        bins = np.floor(readings / (2 * belief.signal_sigma)).astype(int).ravel()
        mass = np.bincount(bins, weights=belief.p.ravel())
        mass = mass[mass > 0]
        return float(-(mass * np.log2(mass)).sum())

    def _learn_from_last_action(self, pos, obs):
        a = self.last_action
        if not a:
            return
        if a["type"] == "move" and obs["message"].startswith("blocked"):
            self.blocked_moves.add((pos, a["dir"]))
        if a["type"] == "scan" and obs["message"].startswith("scanned"):
            self.scanned |= self._dist(pos) <= self.scan_radius

    def _update_phase(self, t, conf):
        new = "explore" if conf < 0.3 else "triangulate" if conf < 0.7 else "close_in"
        note = ""
        if new != self.phase:
            if self.phase is not None:
                self.strategy_changes.append((t, self.phase, new, round(conf, 3)))
                note = f"STRATEGY CHANGE {self.phase} -> {new}. "
            self.phase = new
        return note

    def _commit(self, action, why):
        self.last_action = action
        return action, why

    def _step_toward(self, pos, target):
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        options = []
        if dx:
            options.append((abs(dx), "E" if dx > 0 else "W"))
        if dy:
            options.append((abs(dy), "N" if dy > 0 else "S"))
        for _, d in sorted(options, reverse=True):
            if (pos, d) not in self.blocked_moves:
                return {"type": "move", "dir": d}
        for d in DIRS:  # both useful directions blocked: sidestep
            if (pos, d) not in self.blocked_moves:
                return {"type": "move", "dir": d}
        return {"type": "abort"}

    # ----------------------------------------------------------------- decide
    def decide(self, obs, belief):
        pos = tuple(obs["pos"])
        battery, time_left, t = obs["battery"], obs["time_left"], obs["t"]
        self._learn_from_last_action(pos, obs)
        self.visits[pos] = self.visits.get(pos, 0) + 1

        conf, ent = belief.confidence(), belief.entropy()
        note = self._update_phase(t, conf)
        tag = f"{note}[{self.phase}] conf {conf:.2f}, entropy {ent:.1f} bits, battery {battery}"

        # 1. Sitting on a station with a low battery: top up.
        if (
            pos in self.stations
            and battery < 0.7 * self.max_battery
            and time_left > self.recharge_time + 3
        ):
            return self._commit({"type": "recharge"}, f"{tag}; on a station, recharging")

        # 2. Build and score candidate actions.
        success_w, approach_w = self.PHASES[self.phase]
        ed_now = float((belief.p * self._dist(pos)).sum())  # expected distance to EVE
        cands = []  # (description, action, resulting position, cost, gain)

        for d, (dx, dy) in DIRS.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if not self._in_bounds(nx, ny) or (pos, d) in self.blocked_moves:
                continue
            ed_new = float((belief.p * self._dist((nx, ny))).sum())
            gain = self._reading_info(belief, (nx, ny)) + approach_w * (ed_now - ed_new)
            gain -= 0.15 * self.visits.get((nx, ny), 0)  # discourage pacing back and forth
            cands.append((f"move {d}", {"type": "move", "dir": d}, (nx, ny), self.move_cost, gain))

        dist_here = self._dist(pos)
        disc = dist_here <= self.scan_radius
        p_find = float(belief.p[disc].sum())
        unscanned = float((~self.scanned[disc]).mean())
        mass_near = float(belief.p[dist_here <= self.scan_radius + 4].sum())
        scan_gain = (
            h2(p_find)                                             # info from hit/miss
            + success_w * p_find                                   # reward for finding EVE
            + self.clue_bonus * unscanned * min(1.0, 4 * mass_near)  # chance of new clues
        )
        if scan_gain > 0.02:  # a scan that cannot tell us anything new is never worth its cost
            cands.append(
                (f"scan r{self.scan_radius}", {"type": "scan", "radius": self.scan_radius},
                 pos, self.scan_cost, scan_gain)
            )

        # 3. Safety filter: after acting we must still be able to reach a station.
        safe = [
            c for c in cands
            if battery >= c[3] and battery - c[3] - self._home_cost(c[2]) >= self.margin
        ]

        if safe:
            ranked = sorted(((c[4] / c[3], c) for c in safe), key=lambda s: -s[0])
            self.last_scores = [(c[0], round(s, 3)) for s, c in ranked]
            score, best = ranked[0]
            why = f"{tag}; best: {best[0]} (score {score:.2f})"
            if len(ranked) > 1:
                why += f", next: {ranked[1][1][0]} ({ranked[1][0]:.2f})"
            return self._commit(best[1], why)

        # 4. Nothing safe to do: recharge, head back to a station, or turn back.
        self.last_scores = []
        steps, station = self._home(pos)
        if pos in self.stations:
            if battery < self.max_battery - 1 and time_left > self.recharge_time + 1:
                return self._commit({"type": "recharge"}, f"{tag}; no safe action left, recharging")
            return self._commit({"type": "abort"}, f"{tag}; nothing safe left to do, turning back")
        if battery >= steps * self.move_cost:
            return self._commit(
                self._step_toward(pos, station),
                f"{tag}; battery low, returning to station at {station} ({steps} steps)",
            )
        return self._commit(
            {"type": "abort"}, f"{tag}; cannot reach any station, aborting with current belief"
        )


if __name__ == "__main__":
    # Demo: run one full mission and print each decision. (run.py comes in step 6.)
    import sys

    from belief import Belief
    from env import Env, load_config

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/sample_mission.yaml"
    env = Env(load_config(path))
    belief = Belief(env.public_info())
    agent = Agent(env.public_info())
    obs = env.reset()
    belief.update(None, obs)

    while not env.done:
        action, why = agent.decide(obs, belief)
        obs = env.step(action)
        belief.update(action, obs)
        print(f"t={obs['t']:3d} {str(action):38s} -> pos={obs['pos']} bat={obs['battery']:3d} | {why}")

    print("\nstatus:", obs["status"], "| steps:", obs["t"], "| battery left:", obs["battery"])
    print("strategy changes:", agent.strategy_changes)
    print("true EVE position:", env.ground_truth()["eve"], "| final guess:", belief.best_guess())