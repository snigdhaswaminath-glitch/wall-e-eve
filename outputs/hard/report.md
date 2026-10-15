# Mission Report: hard_mission

**Outcome:** TIME EXPIRED - EVE not found  
**Steps used:** 122 of 150 time units  |  **Battery left:** 13/30  |  **Moves:** 75, **scans:** 40, **recharges:** 7

## Evidence found

- Initial signal: bearing 45 deg, strength 0.2 (used as the prior).
- Strongest signal reading: 0.607 at (5, 7) (step 89).
- Scans that found nothing (ruling out their area): 40.
- Clues discovered: none.

## Inferred location and status

- **Best estimate of EVE's location:** (7, 11)
- Top candidate cells: (7, 11) (99%), (9, 10) (1%), (8, 10) (0%)
- **Status:** likely located
- **Confidence:** 0.99 (probability mass within 2 cells of the estimate); remaining uncertainty 0.08 of 9.81 bits.

## Strategy changes

- Step 2: explore -> triangulate (confidence 0.302)
- Step 5: triangulate -> explore (confidence 0.268)
- Step 7: explore -> triangulate (confidence 0.315)
- Step 14: triangulate -> close_in (confidence 0.786)
- Step 24: close_in -> triangulate (confidence 0.657)
- Step 38: triangulate -> close_in (confidence 0.776)

## Recommended next step

Recharge at the station at (0, 0) (2 steps away), then travel to the best guess (7, 11) and scan there. Battery 13 is not enough for the whole trip (about 31 needed).

## Evaluation (uses hidden ground truth; the agent never saw this)

- True EVE position: (7, 11)
- Error of the agent's estimate: 0.0 cells
- Estimate within 2 cells of the truth: yes
- Clues that existed: 5; clues found: 0
