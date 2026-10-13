# BirdDetector

Frame-level bird presence classification from a balcony webcam.
When the model detects a bird at the feeder, the system captures a photo automatically.

## Status

🚧 Stage 0: project bootstrap.

## Setup

    uv sync
    uv run pre-commit install

## Layout

- `src/bird_detector/` — package code
- `configs/` — run configurations
- `tests/` — pytest suite
- `data/`, `models/`, `logs/` — local artifacts (not in git)
