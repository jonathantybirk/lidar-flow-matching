# LiDAR Flow Matching

Flow matching for predicting what a LiDAR sensor will observe after it moves:

\[
(P_t,\;T_t^{-1}T_q)\longrightarrow P_q
\]

The object remains static. The relative sensor pose is the action.

## Setup

Python 3.14 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run hf auth login
uv run wandb login
```

ShapeNetCore access is available from
[`ShapeNet/ShapeNetCore`](https://huggingface.co/datasets/ShapeNet/ShapeNetCore).

## Data

The current dataset contains buses, cars, motorcycles, and trains. Each object
has 32 training views or 20 evaluation views from a `32 x 32` LiDAR. Splits are
made by object before views are generated.

```bash
uv run python prepare_data.py download 02924116 02958343 03790512 04468005
uv run python prepare_data.py generate vehicles 02924116 02958343 03790512 04468005
uv run python prepare_data.py inspect vehicles 02958343 <MODEL_ID> --view 0 --show
```

## Training

```bash
uv run python train.py --batch-size 8 --workers 0 --run view-flow-local
uv run python review.py runs/view-flow-a100/best.pt
```

`review.py` renders the source mesh, the target scan, the input scan and the
generated scan in the object frame, with hit-mask IoU and return-range error.
Cars are two thirds of the data, so `--synset` is needed to reach the others.

Training has a deliberately unreachable step ceiling and stops from validation
or the job time limit. On DTU HPC:

```bash
mkdir -p runs
bsub < train.lsf
```

Set `run` at the top of `train.lsf` for each job. Architecture details are in
[DESIGN.md](DESIGN.md).
