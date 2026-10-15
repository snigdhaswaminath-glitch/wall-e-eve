"""belief.py - Bayesian belief over EVE's location.

The belief is a probability for every grid cell ("EVE is here"), always
normalized to sum to 1. It starts from a prior built from the initial signal
(bearing + strength) and is reweighted by every new piece of evidence:

  * signal readings  -> a ring of cells whose distance matches the strength
  * clues            -> cells matching the hinted bearing and distance
  * empty scans      -> cells inside the scanned radius are ruled out

It only uses information the agent is allowed to have (env.public_info() and
observations), never the environment's hidden state.
"""
import math

import numpy as np


class Belief:
    def __init__(
        self,
        info,
        signal_sigma=0.03,        # model's tolerance on a signal reading
        clue_bearing_sigma=15.0,  # degrees
        clue_dist_sigma=2.0,      # cells
    ):
        self.w, self.h = info["width"], info["height"]
        self.scale = info["signal_scale"]
        self.start = tuple(info["start"])
        self.signal_sigma = signal_sigma
        self.clue_bearing_sigma = clue_bearing_sigma
        self.clue_dist_sigma = clue_dist_sigma

        self.X, self.Y = np.meshgrid(np.arange(self.w), np.arange(self.h), indexing="ij")
        self.last_signal_pos = None
        self.skipped_updates = 0  # evidence that contradicted everything
        self.p = self._prior(info["initial_signal"])

    # ---------------------------------------------------------------- helpers
    def _dist(self, pos):
        return np.hypot(self.X - pos[0], self.Y - pos[1])

    def _bearing_diff(self, pos, bearing_deg):
        """Signed angle (deg) between 'direction from pos to each cell' and bearing."""
        ang = np.degrees(np.arctan2(self.Y - pos[1], self.X - pos[0]))
        return (ang - bearing_deg + 180.0) % 360.0 - 180.0

    def _apply(self, likelihood):
        """Bayes' rule: posterior is proportional to prior * likelihood."""
        new = self.p * likelihood
        total = new.sum()
        if not np.isfinite(total) or total <= 0:
            self.skipped_updates += 1  # keep the old belief rather than crash
            return
        self.p = new / total

    # ------------------------------------------------------------------ prior
    def _prior(self, sig):
        strength = min(max(sig["strength"], 0.01), 0.999)
        d0 = -self.scale * math.log(strength)  # distance implied by the signal
        dist = self._dist(self.start)
        dth = self._bearing_diff(self.start, sig["bearing_deg"])
        lik = np.exp(-0.5 * ((dist - d0) / (0.2 * d0 + 1.0)) ** 2)
        lik *= np.exp(-0.5 * (dth / 25.0) ** 2)
        lik += 1e-6  # never rule a cell out completely
        lik[self.start] = 0.0  # EVE is not at the start cell
        return lik / lik.sum()

    # ---------------------------------------------------------------- updates
    def update_signal(self, pos, reading):
        expected = np.exp(-self._dist(pos) / self.scale)  # reading if EVE were in each cell
        lik = np.exp(-0.5 * ((expected - reading) / self.signal_sigma) ** 2) + 1e-9
        self._apply(lik)

    def update_clue(self, clue):
        pos = tuple(clue["pos"])
        hint = clue["hint"]
        dist = self._dist(pos)
        lik = np.exp(-0.5 * ((dist - hint["distance_to_eve"]) / self.clue_dist_sigma) ** 2)
        dth = self._bearing_diff(pos, hint["bearing_to_eve"])
        lik *= np.exp(-0.5 * (dth / self.clue_bearing_sigma) ** 2)
        lik += 1e-9
        lik[pos] = 0.0  # a clue is never at EVE's own cell
        self._apply(lik)

    def update_scan_miss(self, pos, radius):
        """A scan that did not find EVE rules out every cell within its radius."""
        self._apply((self._dist(pos) > radius).astype(float))

    def update_scan_hit(self, pos, radius):
        """A scan that found EVE means EVE is somewhere within its radius."""
        self._apply((self._dist(pos) <= radius).astype(float))

    def update(self, action, obs):
        """Call once per observation (including the one from env.reset(), with action=None)."""
        pos = tuple(obs["pos"])
        # Only use a signal reading when the position changed, so standing
        # still does not count the same evidence twice.
        if pos != self.last_signal_pos:
            self.update_signal(pos, obs["signal"])
            self.last_signal_pos = pos
        for clue in obs["new_clues"]:
            self.update_clue(clue)
        if action and action.get("type") == "scan" and obs["message"].startswith("scanned"):
            radius = action.get("radius", 2)  # (the message check means the scan actually ran)
            if obs["eve_found"]:
                self.update_scan_hit(pos, radius)
            else:
                self.update_scan_miss(pos, radius)

    # ---------------------------------------------------------------- queries
    def best_guess(self):
        x, y = np.unravel_index(np.argmax(self.p), self.p.shape)
        return int(x), int(y)

    def entropy(self):
        """Uncertainty in bits. Higher = more unsure."""
        nz = self.p[self.p > 0]
        return float(-(nz * np.log2(nz)).sum())

    def max_entropy(self):
        return math.log2(self.w * self.h)

    def confidence(self, radius=2):
        """Probability mass within `radius` cells of the best guess (0 to 1)."""
        return float(self.p[self._dist(self.best_guess()) <= radius].sum())

    def top(self, n=3):
        flat = np.argsort(self.p, axis=None)[::-1][:n]
        cells = [np.unravel_index(i, self.p.shape) for i in flat]
        return [((int(x), int(y)), round(float(self.p[x, y]), 4)) for x, y in cells]

    def status(self, eve_found=False):
        """Rough status label for the report."""
        if eve_found:
            return "confirmed"
        c = self.confidence()
        if c >= 0.5:
            return "likely located"
        if c >= 0.2:
            return "region narrowed"
        return "unknown"

    def summary(self):
        """Small dict for the trace log."""
        return {
            "best_guess": self.best_guess(),
            "top_cells": self.top(3),
            "entropy_bits": round(self.entropy(), 3),
            "confidence": round(self.confidence(), 3),
        }


if __name__ == "__main__":
    # Smoke test: walk toward the current best guess, scanning now and then,
    # and watch the belief sharpen. (A real agent comes in step 5.)
    import sys

    from env import DIRS, Env, load_config

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/sample_mission.yaml"
    env = Env(load_config(path))
    belief = Belief(env.public_info())
    obs = env.reset()
    belief.update(None, obs)
    print(f"start: entropy {belief.entropy():.2f}/{belief.max_entropy():.2f} bits, "
          f"guess {belief.best_guess()}")

    while not env.done and env.t < 60:
        gx, gy = belief.best_guess()
        px, py = obs["pos"]
        if env.t % 4 == 3:
            action = {"type": "scan", "radius": 2}
        elif abs(gx - px) >= abs(gy - py) and gx != px:
            action = {"type": "move", "dir": "E" if gx > px else "W"}
        elif gy != py:
            action = {"type": "move", "dir": "N" if gy > py else "S"}
        else:
            action = {"type": "scan", "radius": 2}
        obs = env.step(action)
        belief.update(action, obs)
        if env.t % 5 == 0 or env.done:
            print(f"t={env.t:2d} pos={obs['pos']} entropy={belief.entropy():5.2f} "
                  f"guess={belief.best_guess()} conf={belief.confidence():.2f}")

    print("end status:", obs["status"], "| belief status:", belief.status(obs["eve_found"]))
    print("true EVE position:", env.ground_truth()["eve"])
