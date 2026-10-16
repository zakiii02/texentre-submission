# Checkout item counting


## Approach and assumptions

- A pretrained YOLO26n detects people, labeled customer/cashier using their ROIs;
  this does not identify checkout sessions. A separate YOLO26s was fine-tuned for
  two classes: `retail_item` and `carry_bag`. Training ran for 66 epochs, with the
  best checkpoint selected from epoch 46.
- Annotation experiments used multiple vision-language models/tools, including
  Grounding DINO with text prompts, to propose bounding boxes. Auto-annotations
  were reviewed and corrected before training; model proposals were not treated
  as ground truth.
- I have also included a folder for you to check which has sample images i used for training.
- Separate ByteTrack trackers for items/bags, appearance matching and short
  optical-flow tracking reduce identity changes. Counts use distinct identities
  per ROI; brief ambiguous duplicate tracks are excluded and logged.
- Item ROI checks use box center OR bottom-center, plus 20% box coverage.
  Person checks use bottom-center only. An extra crop covers packing outside
  the original detector crop.
- Packing uses bag overlap followed by disappearance. If hands hide insertion,
  a stable item disappearing near a newly detected bag supplies a **low-confidence
  inventory estimate**. Temporary replacement IDs are checked to avoid duplicates.
- Identical/overlapping products can still split or merge identities. Multipacks
  count as one detected unit. Hidden initial contents cannot be counted. Removal
  can resemble packing; insertion and exact bag assignment are not verified.
  Bagged items are a subset, not added to retail totals; bags are counted separately.
  Scanner entry does **not** prove a POS scan or payment
## Run

Tested with Python 3.12 on Linux. Both model weights and the input video are included.

```bash
cd submission
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r req.txt
python3 main.py
```

`DEVICE = 'auto'` uses CUDA when available, otherwise CPU (slower).
For a machine without a desktop display, set `SHOW_WINDOW = False` in `main.py`.
Space pauses; Q/Esc saves and exits early.

## Session and ROIs

Automatic checkout-session detection was **not implemented** (optional in the
assignment). Set `SESSION_START_SECONDS` and `SESSION_END_SECONDS` in `main.py`;
the included example uses **6–60 seconds, center lane only**. Calculations and saved
video cover this interval. Preview playback starts at the beginning. Additional
sessions require changing the times and rerunning; the first three minutes are
not automatically segmented. Waiting customers, companions and overlapping
sessions are not separated automatically.

ROIs were manually drawn and saved in `roi_config.yml`. To edit them:

```bash
python3 draw_roi.py videoplayback.mp4
```

F freezes a frame; click polygon corners, S saves with a name, Q exits.
Names: `CONVEYOR`, `SCANNER`, `BAGGING_AREA`, `Customer_Roi`, `Cashier_Roi`;
append `_2` for the right lane. Comment out a lane's `CONVEYOR` to disable it.
Keep the original 1920×1080 video coordinates.



## Outputs and validation

`example_output/` contains the completed annotated video and JSON results, tracks
and evidence events. New runs save to `outputs/<run-id>/`; timestamps refer to the
source video. `null` means unavailable, not zero.

Manual review found **5 items in bagging and 5 packed**; the example reports **5/5**.
It reports **3 bags**, not independently validated. Raw retail identities still
number **7**, so the overall distinct-item total remains overestimated; conveyor
count **2** is only that ROI's count. Do not sum ROI counts. The model was trained
on this same video, so these results do not establish accuracy on new footage.
