# Mission Report: hard_mission

**Outcome:** ABORTED - agent turned back to stay safe  
**Steps used:** 80 of 150 time units  |  **Battery left:** 7/30  |  **Moves:** 77, **scans:** 0, **recharges:** 2

## Evidence found

- Initial signal: bearing 45 deg, strength 0.2 (used as the prior).
- Strongest signal reading: 0.4961 at (6, 5) (step 41).
- Scans that found nothing (ruling out their area): 0.
- Clues discovered: none.

## Inferred location and status

- **Best estimate of EVE's location:** (7, 11)
- Top candidate cells: (7, 11) (100%), (8, 10) (0%), (6, 11) (0%)
- **Status:** likely located
- **Confidence:** 1.00 (probability mass within 2 cells of the estimate); remaining uncertainty 0.00 of 9.81 bits.

## Strategy changes

- Step 2: explore -> triangulate (confidence 0.302)
- Step 5: triangulate -> explore (confidence 0.268)
- Step 7: explore -> triangulate (confidence 0.315)
- Step 14: triangulate -> close_in (confidence 0.786)
- Step 26: close_in -> triangulate (confidence 0.651)
- Step 27: triangulate -> close_in (confidence 0.726)
- Step 32: close_in -> triangulate (confidence 0.65)
- Step 54: triangulate -> close_in (confidence 0.763)

## Recommended next step

Recharge at the station at (0, 0) (8 steps away), then travel to the best guess (7, 11) and scan there. Battery 7 is not enough for the whole trip (about 25 needed).

## Evaluation (uses hidden ground truth; the agent never saw this)

- True EVE position: (7, 11)
- Error of the agent's estimate: 0.0 cells
- Estimate within 2 cells of the truth: yes
- Clues that existed: 5; clues found: 0
