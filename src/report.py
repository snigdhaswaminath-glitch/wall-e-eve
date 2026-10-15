"""report.py - writes the final mission report (Markdown).

The report is built from the trace and the agent's belief only. The hidden
ground truth is used in one clearly labelled evaluation section at the end.
"""
import math

STATUS_TEXT = {
    "found_eve": "SUCCESS - EVE was found by a scan",
    "out_of_time": "TIME EXPIRED - EVE not found",
    "stranded": "STRANDED - battery ran out away from a station",
    "aborted": "ABORTED - agent turned back to stay safe",
}


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def recommend_next_step(cfg, obs, belief):
    """Plain-language recommendation based on where the belief points and the battery left."""
    pos, battery = tuple(obs["pos"]), obs["battery"]
    guess = belief.best_guess()
    mc, sc = cfg["costs"]["move"], cfg["costs"]["scan"]
    stations = [tuple(s) for s in cfg["recharge_stations"]] or [tuple(cfg["start"])]
    near_pos = min(stations, key=lambda s: _manhattan(pos, s))
    near_guess = min(stations, key=lambda s: _manhattan(guess, s))
    trip = _manhattan(pos, guess) * mc
    need = trip + sc + _manhattan(guess, near_guess) * mc  # go, scan, then get back to a station

    if obs["eve_found"]:
        text = (
            f"EVE is confirmed within scan range of WALL-E's final position {pos} "
            f"(best estimate {guess}). Next move: approach {guess} and make contact."
        )
        if battery < need:
            text += (
                f" Battery is {battery}, so recharge first at the station at "
                f"{near_pos} ({_manhattan(pos, near_pos)} steps away)."
            )
        return text
    if belief.confidence() < 0.2:
        top = ", ".join(str(c) for c, _ in belief.top(3))
        return (
            f"Evidence is still weak (confidence {belief.confidence():.2f}). Recharge if needed "
            f"at {near_pos}, then gather more signal readings around the highest-probability "
            f"cells ({top}) before committing to a trip."
        )
    if battery >= need:
        return (
            f"Travel to the best guess {guess} ({_manhattan(pos, guess)} steps) and scan there; "
            f"battery {battery} covers the trip, a scan, and the return to a station."
        )
    return (
        f"Recharge at the station at {near_pos} ({_manhattan(pos, near_pos)} steps away), then "
        f"travel to the best guess {guess} and scan there. Battery {battery} is not enough "
        f"for the whole trip (about {need} needed)."
    )


def write_report(path, cfg, trace, obs, belief, agent, truth, config_name="mission"):
    status = obs["status"]
    moves = scans = recharges = 0
    for e in trace[1:]:
        kind = e["action"]["type"]
        moves += kind == "move"
        scans += kind == "scan"
        recharges += kind == "recharge" and e["observation"]["message"].startswith("recharged")
    empty_scans = sum(
        1 for e in trace[1:]
        if e["action"]["type"] == "scan"
        and e["observation"]["message"].startswith("scanned")
        and not e["observation"]["eve_found"]
    )

    clues = [(e["t"], c) for e in trace for c in e["observation"]["new_clues"]]
    readings = [(e["observation"]["signal"], e["observation"]["pos"], e["t"]) for e in trace]
    strongest = max(readings)
    init = cfg["initial_signal"]

    guess = belief.best_guess()
    eve_found = obs["eve_found"]
    L = []
    L.append(f"# Mission Report: {config_name}\n")
    L.append(f"**Outcome:** {STATUS_TEXT.get(status, status)}  ")
    L.append(
        f"**Steps used:** {obs['t']} of {cfg['budget']['time']} time units  |  "
        f"**Battery left:** {obs['battery']}/{cfg['budget']['battery']}  |  "
        f"**Moves:** {moves}, **scans:** {scans}, **recharges:** {recharges}\n"
    )

    L.append("## Evidence found\n")
    L.append(
        f"- Initial signal: bearing {init['bearing_deg']} deg, strength {init['strength']} "
        f"(used as the prior)."
    )
    L.append(
        f"- Strongest signal reading: {strongest[0]} at {tuple(strongest[1])} (step {strongest[2]})."
    )
    L.append(f"- Scans that found nothing (ruling out their area): {empty_scans}.")
    if clues:
        L.append(f"- Clues discovered: {len(clues)}")
        for t, c in clues:
            h = c["hint"]
            L.append(
                f"  - Step {t}: clue #{c['id']} at {tuple(c['pos'])} - EVE is roughly "
                f"{h['distance_to_eve']} cells away at bearing {h['bearing_to_eve']} deg."
            )
    else:
        L.append("- Clues discovered: none.")
    L.append("")

    L.append("## Inferred location and status\n")
    L.append(f"- **Best estimate of EVE's location:** {guess}")
    L.append(
        "- Top candidate cells: "
        + ", ".join(f"{c} ({p:.0%})" for c, p in belief.top(3))
    )
    L.append(f"- **Status:** {belief.status(eve_found)}")
    L.append(
        f"- **Confidence:** {belief.confidence():.2f} (probability mass within 2 cells of the "
        f"estimate); remaining uncertainty {belief.entropy():.2f} of {belief.max_entropy():.2f} bits."
    )
    L.append("")

    L.append("## Strategy changes\n")
    if agent.strategy_changes:
        for t, old, new, conf in agent.strategy_changes:
            L.append(f"- Step {t}: {old} -> {new} (confidence {conf})")
    else:
        L.append(f"- None; the agent stayed in the '{agent.phase}' phase.")
    L.append("")

    L.append("## Recommended next step\n")
    L.append(recommend_next_step(cfg, obs, belief))
    L.append("")

    L.append("## Evaluation (uses hidden ground truth; the agent never saw this)\n")
    err = math.hypot(guess[0] - truth["eve"][0], guess[1] - truth["eve"][1])
    L.append(f"- True EVE position: {tuple(truth['eve'])}")
    L.append(f"- Error of the agent's estimate: {err:.1f} cells")
    L.append(f"- Estimate within 2 cells of the truth: {'yes' if err <= 2 else 'no'}")
    L.append(f"- Clues that existed: {len(truth['clues'])}; clues found: {len(clues)}")
    L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))
