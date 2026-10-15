"""viz.py - ASCII visualization of a finished mission.

Produces one text file with:
  1. a map of the path, clues, recharge stations and strategy-change points
  2. an action timeline with the strategy phase underneath each step
  3. belief heatmaps at the start, just before each strategy change, and at the end

This is a post-mission view for humans. It draws the true obstacles, EVE and
unfound clues (which the agent never saw), so it is clearly marked as such.
"""
import numpy as np

HEAT = ".:-=+*#%@"  # faint -> strong belief
PHASE_CHAR = {"explore": "e", "triangulate": "t", "close_in": "c", None: " "}


def _axis_lines(w):
    tens = "   " + " ".join(str(x // 10) if x % 10 == 0 else " " for x in range(w))
    ones = "   " + " ".join(str(x % 10) for x in range(w))
    return [tens, ones]


def _grid_lines(grid, w, h):
    """grid[y][x] -> text lines with y increasing upward (north is up)."""
    rows = [f"{y:2d} " + " ".join(grid[y]) for y in range(h - 1, -1, -1)]
    return _axis_lines(w) + rows + _axis_lines(w)


def render_path_map(env, trace, changes):
    w, h = env.w, env.h
    truth = env.ground_truth()
    grid = [["." for _ in range(w)] for _ in range(h)]

    for x in range(w):
        for y in range(h):
            if env.blocked[x][y]:
                grid[y][x] = "#"
    for e in trace:
        x, y = e["observation"]["pos"]
        grid[y][x] = "*"
    for x, y in env.stations:
        grid[y][x] = "R"
    sx, sy = env.start
    grid[sy][sx] = "S"

    found_ids = {c["id"] for e in trace for c in e["observation"]["new_clues"]}
    for c in truth["clues"]:
        x, y = c["pos"]
        grid[y][x] = "C" if c["id"] in found_ids else "c"

    legend_changes = []
    for k, (t, old, new, conf) in enumerate(changes, 1):
        x, y = trace[t]["observation"]["pos"]
        grid[y][x] = str(k) if k <= 9 else "+"
        legend_changes.append(f"  {k}: step {t}, {old} -> {new} at ({x}, {y}), confidence {conf}")

    ex, ey = truth["eve"]
    grid[ey][ex] = "E"
    wx, wy = trace[-1]["observation"]["pos"]
    grid[wy][wx] = "W"

    lines = ["PATH MAP (north is up)", ""]
    lines += _grid_lines(grid, w, h)
    lines += [
        "",
        "Legend: S start | R recharge station | * cells WALL-E visited | W WALL-E at the end",
        "        C clue found | c clue never found | # obstacle | E EVE (true position)",
        "        1-9 point where the strategy changed (listed below)",
        "(Obstacles, E and unfound clues were hidden from the agent; shown here for evaluation.)",
    ]
    if legend_changes:
        lines += ["", "Strategy changes:"] + legend_changes
    else:
        lines += ["", "Strategy changes: none"]
    return lines


def render_timeline(trace, width=50):
    acts, phases = [], []
    arrows = {"N": "^", "S": "v", "E": ">", "W": "<"}
    for e in trace[1:]:
        a, obs = e["action"], e["observation"]
        kind = a["type"]
        if kind == "move":
            ch = "!" if obs["message"].startswith("blocked") else arrows[a["dir"]]
        elif kind == "scan":
            ch = "X" if obs["eve_found"] else "o"
        elif kind == "recharge":
            ch = "R"
        else:
            ch = "A"
        acts.append(ch)
        phases.append(PHASE_CHAR.get(e["phase"], "?"))

    lines = [
        "ACTION TIMELINE",
        "",
        "Actions: ^ v > < move N/S/E/W | ! blocked move | o scan (nothing) | X scan found EVE",
        "         R recharge | A abort",
        "Phases:  e explore | t triangulate | c close_in",
        "",
    ]
    for start in range(0, len(acts), width):
        chunk = slice(start, start + width)
        label = f"{start + 1:4d}-{min(start + width, len(acts)):<4d}"
        lines.append(f"steps {label} {''.join(acts[chunk])}")
        lines.append(f"{'':16s}{''.join(phases[chunk])}")
    return lines


def render_belief_panel(p, w, h, pos, eve, header):
    pmax = float(p.max())
    grid = [[" " for _ in range(w)] for _ in range(h)]
    for x in range(w):
        for y in range(h):
            r = float(p[x, y]) / pmax if pmax > 0 else 0.0
            if r >= 0.005:
                grid[y][x] = HEAT[min(len(HEAT) - 1, int((r ** 0.5) * len(HEAT)))]
    grid[eve[1]][eve[0]] = "E"
    grid[pos[1]][pos[0]] = "W"

    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    bx, by = np.unravel_index(int(np.argmax(p)), p.shape)
    near = sum(
        float(p[i, j]) for i in range(w) for j in range(h)
        if (i - bx) ** 2 + (j - by) ** 2 <= 4
    )
    lines = [header, f"  entropy {entropy:.2f} bits | best guess ({int(bx)}, {int(by)}) | confidence {near:.2f}", ""]
    lines += _grid_lines(grid, w, h)
    return lines


def write_viz(path, env, trace, changes, snapshots):
    """Write the full visualization to `path` and return the text."""
    out = ["ASCII MISSION VISUALIZATION", "=" * 27, ""]
    out += render_path_map(env, trace, changes)
    out += ["", "=" * 27, ""]
    out += render_timeline(trace)
    out += ["", "=" * 27, "", "BELIEF OVER TIME",
            "Shading: faint '.' to strong '@' (scaled to each panel's peak, blank = ruled out)",
            "W = WALL-E at that moment | E = true EVE position (evaluation only)", ""]

    change_at = {t: (k, old, new) for k, (t, old, new, _c) in enumerate(changes, 1)}
    final_t = trace[-1]["t"]
    for t in sorted(snapshots):
        if t == 0:
            header = "Start (belief from the initial signal)"
        elif t in change_at:
            k, old, new = change_at[t]
            header = f"Step {t}: just before strategy change {k} ({old} -> {new})"
        elif t == final_t:
            header = f"Final (step {t})"
        else:
            header = f"Step {t}"
        pos = trace[t]["observation"]["pos"]
        out += render_belief_panel(snapshots[t], env.w, env.h, pos, env.ground_truth()["eve"], header)
        out.append("")

    text = "\n".join(out) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text
