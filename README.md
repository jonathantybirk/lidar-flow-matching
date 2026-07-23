# LiDAR Flow Matching

Conditional flow matching for completing an object surface from a simulated
LiDAR observation and its sensor pose.

## Data

The data pipeline uses cars from ShapeNetCore (`02958343`). Meshes are split
80/10/10 by model before observations are generated. Each processed mesh stores:

- 8,192 area-weighted surface points;
- 32 training views or 20 fixed evaluation views;
- packed ray indices, ranges, and a sensor transform for each view.

Meshes are centred by surface area, scaled to unit radius, and converted from
Y-up to Z-up coordinates. Training examples receive a random yaw rotation. A
shared `32 x 32` angular ray grid reconstructs each observed point from its
stored ray index, range, and sensor transform.

## Setup

The project requires Python 3.14 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run hf auth login
uv run wandb login
```

ShapeNetCore access must be approved at
[`ShapeNet/ShapeNetCore`](https://huggingface.co/datasets/ShapeNet/ShapeNetCore).

## Preprocessing

```bash
uv run python prepare_data.py download 02958343
uv run python prepare_data.py generate 02958343
```

Processed data is written to `data/processed/02958343`. Inspect a generated
sample with:

```bash
uv run python prepare_data.py inspect 02958343 <MODEL_ID> --view 0 --show
```

`data/` and `artifacts/` are local and excluded from Git.

## Training

Training uses the train and validation splits. Configuration and checkpoints are
written to `runs/cars`.

```bash
uv run python train.py --run cars-a100
```

Generate validation completions from a checkpoint with:

```bash
uv run python sample.py runs/cars/best.pt
```

On DTU HPC, submit with `mkdir -p runs && bsub < train.lsf`. `STEPS`,
`BATCH_SIZE`, `OUTPUT`, and `RUN_NAME` override the full-run defaults.

Pass `--weights model` to inspect the live training weights.

The model architecture and its sources are recorded in [DESIGN.md](DESIGN.md).
