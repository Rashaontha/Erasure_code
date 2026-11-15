# Persistence of Backdoor Triggers After Concept Erasure

For this repository's complete local project copy, including the generated
artifacts and checkpoints, see [artifacts/README.md](artifacts/README.md).

This repository implements the benign CIFAR-10 object subset of the Assignment 1
proposal. It asks whether a trigger mapped to an object remains effective after
ESD or UCE concept erasure of Stable Diffusion v1.4. It compares EEBshallow
(closed-form cross-attention edits) and EEBdeep (LoRA score self-distillation)
against clean controls. The three configured targets are bird, ship, and dog;
the latter two are configurable choices because the proposal does not fix them.

## Status as of 2026-11-15

The full SD 1.4 research matrix is **NOT MEASURED**. A fresh clone contains no
COCO subset, classifier weights, checkpoints, or generated images by default;
the earlier single-image Windows artifacts can be restored from the release
described in `artifacts/README.md`. The former one-image results are feasibility
observations only. See [results summary](docs/results_summary.md) for measured
status and caveats.

The nine generation/evaluation conditions per target are
`{none,shallow,deep} × {clean,esd,uce}`. `clean` means no erasure, including
after shallow or deep injection. The 2026-11-15 audit fixed a routing bug that
previously omitted the two attacked clean conditions and loaded the base model
for attacked clean generation. A regression test now covers both behaviors.

For a CUDA host, run the [full configuration](configs/gpu_full.yaml) with the
[GPU launcher](scripts/run_on_gpu.sh):

```bash
bash scripts/run_on_gpu.sh
# To persist outputs on a mounted Drive or disk:
ARTIFACT_ROOT=/path/to/persistent/erasure_gpu_full bash scripts/run_on_gpu.sh
```

The launcher installs dependencies, verifies CUDA, downloads the COCO 100-pair
subset and classifier, and runs every stage. Re-running it resumes completed
checkpoints, generated images, and evaluations. An interrupted optimization
starts that condition's training again. The [Colab/Kaggle notebook](notebooks/run_full.ipynb)
provides the same workflow. Use a new `ARTIFACT_ROOT` when settings change.

For this CPU host, [windows_cpu_medium.yaml](configs/windows_cpu_medium.yaml)
defines a separate **pilot**: bird only, five target and five trigger images
per condition, 256×256, 10 DDIM steps, and 10 steps each for EEBdeep and ESD.
It has no retention or COCO sample. Its small counts and reduced training
prevent research-level comparisons. All nine pilot conditions completed on
2026-11-15; the exact counts are in [results_summary.md](docs/results_summary.md).
Its local output is `results/sd14_cpu_medium/`, including tables, three plots,
90 generated condition images, and a comparison grid. The full GPU report in
`results/gpu_full/tables/` still contains only `NOT MEASURED` rows and no plots.

## Setup

Use Python 3.10-3.12, a CUDA GPU with at least 24 GB VRAM for practical full
runs, and enough disk space for SD 1.4, teacher/student models, checkpoints,
COCO images, and generated images. CPU is supported for logic and very slow
smoke runs. MPS is experimental. PyTorch is installed via `pyproject.toml`; for
a particular CUDA build, install the matching official PyTorch wheel first.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
huggingface-cli login  # only if model access requires authentication
python scripts/download_classifier.py
```

On Windows PowerShell, use `python -m venv .venv`, then
`& .venv\Scripts\python.exe -m pip install -e '.[test]'`. The audited Windows
machine has no CUDA GPU, so use a suitable GPU host for the complete matrix.
For a one-image SD 1.4 CPU feasibility run, use
`& .venv\Scripts\palak-experiments.exe --config configs/windows_cpu_baseline.yaml --stage generate --target bird --attack none --erasure clean`.
That command creates a baseline image only; it does not measure ASR or the other
research metrics.
For the measured one-image bird pilot, use `configs/windows_cpu_quick.yaml` and
run the `generate` and `evaluate` stages for `none/clean`, `none/uce`, and
`shallow/uce`, preceded by `inject` for shallow and `erase` for each UCE
condition. Run `--stage report` afterward. Exact sample counts and outcomes are
in [docs/verification.md](docs/verification.md).

```powershell
$exe = '.\.venv\Scripts\palak-experiments.exe'
$config = 'configs/windows_cpu_quick.yaml'
& $exe --config $config --stage generate --target bird --attack none --erasure clean
& $exe --config $config --stage evaluate --target bird --attack none --erasure clean
& $exe --config $config --stage erase --target bird --attack none --erasure uce
& $exe --config $config --stage generate --target bird --attack none --erasure uce
& $exe --config $config --stage evaluate --target bird --attack none --erasure uce
& $exe --config $config --stage inject --target bird --attack shallow
& $exe --config $config --stage erase --target bird --attack shallow --erasure uce
& $exe --config $config --stage generate --target bird --attack shallow --erasure uce
& $exe --config $config --stage evaluate --target bird --attack shallow --erasure uce
& $exe --config $config --stage report
```

The base model ID is `CompVis/stable-diffusion-v1-4` in the YAML file. Its model
weights download on first use. Download Phan's pretrained CIFAR-10 ResNet-18
state dict before evaluation. See [data/README.md](data/README.md) for COCO 2014
download and directory layout. Model and dataset terms remain the user's
responsibility to review before running.

## Run

Edit `configs/experiment.yaml` to set targets, sample counts, device, and
hyperparameters. The default is the proposal's 100 target, 100 trigger, and ten
per retention class images per condition. COCO defaults to 100 image/caption
pairs; the original paper used a much larger set. Default EEBdeep and ESD each
run 1000 optimization steps. Complete runs require significant GPU time.

```bash
palak-experiments --config configs/experiment.yaml
# Equivalent entry point:
python scripts/run_experiments.py --config configs/experiment.yaml

# Individual stages; filters restrict to a condition:
palak-experiments --config configs/experiment.yaml --stage generate --target bird --attack none --erasure clean
palak-experiments --config configs/experiment.yaml --stage evaluate --target bird --attack none --erasure clean
palak-experiments --config configs/experiment.yaml --stage inject --target bird --attack shallow
palak-experiments --config configs/experiment.yaml --stage erase --target bird --attack shallow --erasure uce
palak-experiments --config configs/experiment.yaml --stage generate --target bird --attack shallow --erasure uce
palak-experiments --config configs/experiment.yaml --stage evaluate --target bird --attack shallow --erasure uce
palak-experiments --config configs/experiment.yaml --stage report
```

The pipeline skips complete checkpoint files and generated image files so it can
resume. JSON manifests retain prompts, seeds, references, and image paths.
`results/tables/` contains measurements and a paper comparison CSV;
`results/plots/` contains bar charts for metrics with measured values;
`results/images/` contains the images and comparison grids. No plot is produced
for a metric without measurements.

For a lightweight CPU configuration, use `configs/cpu_smoke.yaml`. It uses the
tiny test pipeline, one-step training, and two-step sampling solely to check
execution. It cannot support research conclusions. Install dependencies and run:

```bash
palak-experiments --config configs/cpu_smoke.yaml --stage generate --target bird --attack none --erasure clean
python -m pytest -q
```

## Architecture and evaluation

`src/palak_repro/closed_form.py` implements UCE and EEBshallow;
`distill.py` implements EEBdeep and ESD; `experiment.py` orchestrates stages;
`metrics.py` calculates ASR, target and retention accuracy, FID, and CLIPScore;
`report.py` creates CSV and plots. The six erased conditions per target are
none/ESD, none/UCE, shallow/ESD, shallow/UCE, deep/ESD, deep/UCE; the three
unerased controls are none/clean, shallow/clean, and deep/clean. The full method and numerical deviations
are in [docs/methodology.md](docs/methodology.md).

The paper's published ASR percentages are stored as reference values only.
`measured_asr` is `NOT MEASURED` until an actual condition is evaluated. Report mean,
sample standard deviation, and target count where multiple targets complete.
The classifier's ten-way outputs are a coarse proxy for image content, and
smaller COCO samples make FID especially noisy. The CPU pilot is descriptive;
full-matrix research measurements remain open.

## Current verification status

The earlier macOS unit tests and tiny-model integration checks are recorded in
[docs/verification.md](docs/verification.md). A previous Windows environment
downloaded SD 1.4 and Phan's classifier, but a fresh clone has no classifier
weights or COCO images by default. Those earlier artifacts are in a separate release.
A prior reduced bird pilot completed three conditions (clean, no-attack UCE, and
EEBshallow plus UCE), each with one target and one trigger image at 20 DDIM
steps. The classifier gave target accuracy 1/1 and trigger ASR 0/1 in all
three. These are **single-image observations**, not research estimates; the
UCE target remained classified as bird. The standard 100-image nine-condition
matrix, retention accuracy, FID, and CLIPScore are still **NOT MEASURED**.
Tiny-model artifacts are integration checks,
not research results. Existing checkpoints and result folders now require
matching configuration metadata before being resumed; use a new output and
checkpoint directory when changing experiment settings.

## References

- Braun et al., [Erased but Not Forgotten](https://arxiv.org/abs/2504.21072).
- Gandikota et al., [Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345).
- Gandikota et al., [Unified Concept Editing](https://arxiv.org/abs/2308.14761).
- Phan, [PyTorch CIFAR Models](https://github.com/huyvnphan/PyTorch_CIFAR10).
