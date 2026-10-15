"""run.py - runs one mission end to end.

Usage:  python src/run.py configs/sample_mission.yaml [--out outputs]

Writes:
  <out>/trace.jsonl   one JSON line per step (action, observation, belief, reasoning)
  <out>/report.md     the final mission report
  <out>/viz.txt       ASCII map of the path, timeline and belief heatmaps

Add --show to also print the visualization in the terminal.
"""
import argparse
import json
import os

from agent import Agent
from belief import Belief
from env import Env, load_config
from report import write_report
from viz import write_viz

OBS_KEYS = ("pos", "battery", "time_left", "signal", "new_clues", "eve_found", "status", "message")


def make_entry(obs, action, why, belief, agent):
    """One trace line. The belief shown is AFTER absorbing this step's observation."""
    return {
        "t": obs["t"],
        "action": action,
        "observation": {k: obs[k] for k in OBS_KEYS},
        "belief": belief.summary(),
        "phase": agent.phase,
        "reasoning": why,
        "candidate_scores": agent.last_scores,
    }


def run_mission(cfg, verbose=True):
    """Run one mission. Returns (env, belief, agent, trace, final_obs, snapshots).

    `snapshots` maps step -> copy of the belief grid at the start, just before each
    strategy change, and at the end (used by the visualization).
    """
    env = Env(cfg)
    # Tell the belief model how noisy the signal is (never less than its default tolerance).
    belief = Belief(env.public_info(), signal_sigma=max(0.03, cfg.get("signal_noise", 0.0)))
    agent = Agent(env.public_info())

    obs = env.reset()
    belief.update(None, obs)
    trace = [make_entry(obs, None, "mission start", belief, agent)]
    snapshots = {0: belief.p.copy()}
    n_changes = 0

    while not env.done:
        action, why = agent.decide(obs, belief)
        if len(agent.strategy_changes) > n_changes:  # the phase just changed
            n_changes = len(agent.strategy_changes)
            snapshots[agent.strategy_changes[-1][0]] = belief.p.copy()
        obs = env.step(action)
        belief.update(action, obs)
        trace.append(make_entry(obs, action, why, belief, agent))
        if verbose:
            print(f"t={obs['t']:3d} {action['type']:8s} pos={obs['pos']} "
                  f"battery={obs['battery']:3d} conf={belief.confidence():.2f}")
    snapshots[obs["t"]] = belief.p.copy()
    return env, belief, agent, trace, obs, snapshots


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    parser = argparse.ArgumentParser(description="Run a WALL-E / EVE mission.")
    parser.add_argument("config", help="path to the mission YAML, e.g. configs/sample_mission.yaml")
    parser.add_argument("--out", default="outputs", help="folder for trace.jsonl, report.md and viz.txt")
    parser.add_argument("--show", action="store_true", help="also print the ASCII visualization")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env, belief, agent, trace, obs, snapshots = run_mission(cfg)

    os.makedirs(args.out, exist_ok=True)
    trace_path = os.path.join(args.out, "trace.jsonl")
    with open(trace_path, "w") as f:
        for entry in trace:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    report_path = os.path.join(args.out, "report.md")
    name = os.path.splitext(os.path.basename(args.config))[0]
    write_report(report_path, cfg, trace, obs, belief, agent, env.ground_truth(), config_name=name)

    viz_path = os.path.join(args.out, "viz.txt")
    viz_text = write_viz(viz_path, env, trace, agent.strategy_changes, snapshots)
    if args.show:
        print("\n" + viz_text)

    print(f"\nStatus: {obs['status']} after {obs['t']} steps. Best guess for EVE: {belief.best_guess()}")
    print(f"Wrote {trace_path}, {report_path} and {viz_path}")


if __name__ == "__main__":
    main()
