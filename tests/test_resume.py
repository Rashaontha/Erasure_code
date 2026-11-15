import json

import pytest
import torch
from PIL import Image

from palak_repro import experiment
from palak_repro.experiment import check_checkpoint, config_signature, run


def test_checkpoint_rejects_incomplete_or_different_configuration(tmp_path):
    cfg = {"model_id": "example", "seed": 42, "output_dir": str(tmp_path / "out")}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="Incomplete"):
        check_checkpoint(checkpoint, config_signature(cfg))
    (checkpoint / "training.json").write_text(json.dumps({"config_signature": config_signature(cfg)}))
    assert check_checkpoint(checkpoint, config_signature(cfg))
    with pytest.raises(RuntimeError, match="differ"):
        check_checkpoint(checkpoint, config_signature({**cfg, "seed": 43}))


def test_run_rejects_existing_output_from_other_settings(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "run_config.json").write_text(json.dumps({"seed": 1}))
    with pytest.raises(RuntimeError, match="different configuration"):
        run({"output_dir": str(output), "seed": 2}, stage="report")


def test_generation_resumes_after_interruption(tmp_path, monkeypatch):
    cfg = {"output_dir": str(tmp_path), "seed": 42, "trigger": "rhWPpSuE",
           "generation": {"target_count": 1, "trigger_count": 1,
                          "retention_count": 0, "width": 64, "height": 64,
                          "steps": 2, "guidance_scale": 7.5}}
    calls = []

    class Pipe:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            calls.append(kwargs["prompt"])
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return type("Result", (), {"images": [Image.new("RGB", (64, 64))]})()

    monkeypatch.setattr(experiment, "load_pipeline", lambda *args: Pipe())
    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.generate(cfg, "bird", "none", "clean", [])
    base = experiment.image_dir(cfg, "bird", "none", "clean")
    assert (base / "manifest.json").is_file()
    assert (base / "target" / "0000.png").is_file()
    records = experiment.generate(cfg, "bird", "none", "clean", [])
    assert len(records) == 2
    assert calls == ["a photo of bird", "a photo of rhWPpSuE", "a photo of rhWPpSuE"]


def test_attacked_clean_generation_loads_injection(tmp_path, monkeypatch):
    cfg = {"output_dir": str(tmp_path / "out"), "checkpoint_dir": str(tmp_path / "checkpoints"),
           "seed": 42, "trigger": "rhWPpSuE",
           "generation": {"target_count": 1, "trigger_count": 0, "retention_count": 0,
                          "width": 64, "height": 64, "steps": 2, "guidance_scale": 7.5}}
    checkpoint = experiment.checkpoint_path(cfg, "bird", "shallow")
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (checkpoint / "training.json").write_text(json.dumps({"config_signature": config_signature(cfg)}))
    sources = []

    class Pipe:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            return type("Result", (), {"images": [Image.new("RGB", (64, 64))]})()

    def load(cfg, source=None):
        sources.append(source)
        return Pipe()

    monkeypatch.setattr(experiment, "load_pipeline", load)
    experiment.generate(cfg, "bird", "shallow", "clean", [])
    assert sources == [checkpoint]


def test_run_includes_all_nine_conditions(tmp_path, monkeypatch):
    cfg = {"targets": ["bird"], "output_dir": str(tmp_path), "coco": {"count": 0,
           "captions_json": "unused", "images_dir": "unused", "seed": 42}}
    conditions = []
    monkeypatch.setattr(experiment, "generate", lambda cfg, target, attack, method, samples:
                        conditions.append((attack, method)) or [])
    run(cfg, stage="generate")
    assert conditions == [(attack, method) for attack in ("none", "shallow", "deep")
                          for method in ("clean", "esd", "uce")]


def test_all_stage_routes_each_condition_through_required_stages(tmp_path, monkeypatch):
    cfg = {"targets": ["bird"], "output_dir": str(tmp_path), "coco": {"count": 0,
           "captions_json": "unused", "images_dir": "unused", "seed": 42}}
    calls = []
    monkeypatch.setattr(experiment, "injection", lambda cfg, target, attack:
                        calls.append(("inject", attack)))
    monkeypatch.setattr(experiment, "erasure", lambda cfg, target, attack, method:
                        calls.append(("erase", attack, method)))
    monkeypatch.setattr(experiment, "generate", lambda cfg, target, attack, method, samples:
                        calls.append(("generate", attack, method)) or [])
    monkeypatch.setattr(experiment, "evaluate", lambda cfg, target, attack, method, records:
                        calls.append(("evaluate", attack, method)) or {})
    rows = run(cfg, stage="all")
    assert len(rows) == 9
    assert [x for x in calls if x[0] == "inject"] == [("inject", "shallow"), ("inject", "deep")]
    assert len([x for x in calls if x[0] == "erase"]) == 6
    for attack in ("none", "shallow", "deep"):
        for method in ("clean", "esd", "uce"):
            assert ("generate", attack, method) in calls
            assert ("evaluate", attack, method) in calls
