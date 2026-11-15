#!/usr/bin/env bash
# Run from the repository root. Re-running resumes completed checkpoints, images, and evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install -e '.[test]'
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required for configs/gpu_full.yaml")
print("CUDA device:", torch.cuda.get_device_name(0))
PY

"$PYTHON_BIN" scripts/download_coco_subset.py --root data --count 100 --seed 42
if [[ ! -s data/classifier/resnet18.pt ]]; then
    "$PYTHON_BIN" scripts/download_classifier.py
fi

CONFIG=configs/gpu_full.yaml
if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
    mkdir -p "$ARTIFACT_ROOT"
    CONFIG="$ARTIFACT_ROOT/gpu_full.yaml"
    "$PYTHON_BIN" - "$CONFIG" "$ARTIFACT_ROOT" <<'PY'
import pathlib, sys, yaml
cfg = yaml.safe_load(pathlib.Path('configs/gpu_full.yaml').read_text())
root = pathlib.Path(sys.argv[2]).expanduser().resolve()
cfg['output_dir'] = str(root / 'results')
cfg['checkpoint_dir'] = str(root / 'checkpoints')
pathlib.Path(sys.argv[1]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
fi

"$PYTHON_BIN" scripts/run_experiments.py --config "$CONFIG" --stage all
