# Mission Report: sample_mission

**Outcome:** SUCCESS - EVE was found by a scan  
**Steps used:** 7 of 150 time units  |  **Battery left:** 91/100  |  **Moves:** 6, **scans:** 1, **recharges:** 0

## Evidence found

- Initial signal: bearing 45 deg, strength 0.6 (used as the prior).
- Strongest signal reading: 0.7659 at (4, 2) (step 7).
- Scans that found nothing (ruling out their area): 0.
- Clues discovered: none.

## Inferred location and status

- **Best estimate of EVE's location:** (4, 6)
- Top candidate cells: (4, 6) (78%), (6, 5) (18%), (7, 4) (3%)
- **Status:** confirmed
- **Confidence:** 0.79 (probability mass within 2 cells of the estimate); remaining uncertainty 0.98 of 8.64 bits.

## Strategy changes

- Step 4: triangulate -> close_in (confidence 0.789)

## Recommended next step

EVE is confirmed within scan range of WALL-E's final position (4, 2) (best estimate (4, 6)). Next move: approach (4, 6) and make contact.

## Evaluation (uses hidden ground truth; the agent never saw this)

- True EVE position: (4, 6)
- Error of the agent's estimate: 0.0 cells
- Estimate within 2 cells of the truth: yes
- Clues that existed: 5; clues found: 0
