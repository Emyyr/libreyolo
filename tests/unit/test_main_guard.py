"""Unguarded ``train()``/``val()`` scripts under the spawn/forkserver start methods."""

from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from libreyolo.models.base.model import _wrap_train_with_cfg
from libreyolo.utils import main_guard

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_LOGGER = "libreyolo.utils.main_guard"


@pytest.fixture(autouse=True)
def _propagate_guard_warnings(monkeypatch):
    # CLI-style logging setup stops libreyolo records at the package logger;
    # let them reach caplog's root handler here.
    monkeypatch.setattr(logging.getLogger("libreyolo"), "propagate", True)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def _guard_warnings(caplog):
    return [r for r in caplog.records if r.name == GUARD_LOGGER]


# ---------------------------------------------------------------------------
# AST detection of the running __main__ statement
# ---------------------------------------------------------------------------


def _run_as_main(tmp_path, source, *, file=True, filename=None):
    """Run ``source`` as a ``__main__`` script; ``probe()`` records the verdict."""
    path = tmp_path / "script.py"
    source = textwrap.dedent(source)
    if file:
        path.write_text(source, encoding="utf-8")
    seen = []
    namespace = {
        "__name__": "__main__",
        "probe": lambda: seen.append(main_guard.main_script_reruns_in_workers()),
    }
    if filename is None:
        namespace["__file__"] = str(path)
    exec(compile(source, filename or str(path), "exec"), namespace)  # noqa: S102 - test script runs as __main__
    assert len(seen) == 1
    return seen[0]


@pytest.fixture
def spawn(monkeypatch):
    monkeypatch.setattr(main_guard, "effective_start_method", lambda: "spawn")


def test_unguarded_top_level_call_is_detected(tmp_path, spawn):
    assert _run_as_main(tmp_path, "probe()\n") is True


def test_unguarded_call_through_a_function_is_detected(tmp_path, spawn):
    source = """
        import logging
        log = logging.getLogger(__name__)

        def main():
            probe()

        main()
    """
    assert _run_as_main(tmp_path, source) is True


def test_guarded_call_is_not_detected(tmp_path, spawn):
    source = """
        if __name__ == "__main__":
            probe()
    """
    assert _run_as_main(tmp_path, source) is False


def test_guarded_main_function_is_not_detected(tmp_path, spawn):
    source = """
        def main():
            probe()

        if __name__ == "__main__":
            main()
    """
    assert _run_as_main(tmp_path, source) is False


@pytest.mark.parametrize(
    "source",
    [
        # Early exit: guarded in effect, but not in the standard form.
        """
        import sys
        if __name__ != "__main__":
            sys.exit()
        probe()
        """,
        """
        if "__main__" == __name__ or False:
            probe()
        """,
        """
        if __name__ != "__main__":
            pass
        else:
            probe()
        """,
    ],
    ids=["early-exit", "unusual-guard", "inverted-guard"],
)
def test_unusual_guards_are_inconclusive(tmp_path, spawn, source):
    assert _run_as_main(tmp_path, source) is False


@pytest.mark.parametrize(
    "source",
    [
        """
        if True:
            if __name__ == "__main__":
                probe()
        """,
        """
        import sys
        try:
            if __name__ != "__main__":
                sys.exit()
        except ImportError:
            pass
        probe()
        """,
        """
        import sys
        try:
            if __name__ != "__main__":
                sys.exit()
            probe()
        finally:
            pass
        """,
        """
        import sys
        __name__ != "__main__" and sys.exit()
        probe()
        """,
        """
        if __name__ == "__main__":
            setup = 1
        else:
            setup = 2
        probe()
        """,
        """
        if __name__ == "__main__":
            setup = 1
        else:
            raise SystemExit
        probe()
        """,
        """
        DEBUG = True
        if __name__ == "__main__" and DEBUG:
            setup = 1
        probe()
        """,
        """
        assert __name__ == "__main__"
        probe()
        """,
    ],
    ids=[
        "nested-guard",
        "early-exit-in-try",
        "early-exit-in-same-try",
        "bare-expression-exit",
        "worker-branch-not-empty",
        "else-branch-early-exit",
        "compound-test",
        "assert",
    ],
)
def test_guards_and_early_exits_anywhere_are_not_unguarded(tmp_path, spawn, source):
    assert _run_as_main(tmp_path, source) is False


@pytest.mark.parametrize(
    "source",
    [
        # Main-only setup before an unguarded call does not guard the call.
        """
        if __name__ == "__main__":
            setup = 1
        probe()
        """,
        """
        if __name__ == "__main__":
            print(__name__)
        probe()
        """,
        """
        if __name__ != "__main__":
            pass
        probe()
        """,
        """
        if __name__ == "__main__":
            a = 1
        b = 2
        if "__main__" == __name__:
            c = 3
        probe()
        """,
        """
        def helper():
            if __name__ != "__main__":
                raise SystemExit

        probe()
        """,
        """
        probe()
        if __name__ != "__main__":
            raise SystemExit
        """,
    ],
    ids=[
        "guarded-setup-then-call",
        "name-use-in-main-branch",
        "empty-worker-branch",
        "two-passed-guards",
        "name-test-in-function-body",
        "exit-after-the-call",
    ],
)
def test_guards_a_worker_passes_through_do_not_hide_the_call(tmp_path, spawn, source):
    assert _run_as_main(tmp_path, source) is True


def test_missing_source_file_is_inconclusive(tmp_path, spawn):
    assert _run_as_main(tmp_path, "probe()\n", file=False) is False


@pytest.mark.parametrize("filename", ["<string>", "<stdin>"])
def test_code_without_a_script_file_is_inconclusive(tmp_path, spawn, filename):
    # ``python -c``, stdin and notebook cells have no ``__file__``.
    assert _run_as_main(tmp_path, "probe()\n", filename=filename) is False


def test_frozen_application_is_inconclusive(tmp_path, spawn, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert _run_as_main(tmp_path, "probe()\n") is False


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_reimporting_start_methods_are_checked(tmp_path, monkeypatch, method):
    monkeypatch.setattr(main_guard, "effective_start_method", lambda: method)
    assert _run_as_main(tmp_path, "probe()\n") is True


def test_fork_start_method_skips_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(main_guard, "effective_start_method", lambda: "fork")
    assert _run_as_main(tmp_path, "probe()\n") is False


def test_effective_start_method_does_not_fix_the_context(monkeypatch):
    calls = []

    def fake_get_start_method(allow_none=False):
        calls.append(allow_none)

    monkeypatch.setattr(multiprocessing, "get_start_method", fake_get_start_method)
    monkeypatch.setattr(
        multiprocessing, "get_all_start_methods", lambda: ["forkserver", "spawn"]
    )
    assert main_guard.effective_start_method() == "forkserver"
    assert calls == [True]


# ---------------------------------------------------------------------------
# resolve_dataloader_workers
# ---------------------------------------------------------------------------


@pytest.fixture
def reruns(monkeypatch):
    calls = []

    def fake():
        calls.append(True)
        return True

    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", fake)
    monkeypatch.setattr(main_guard, "effective_start_method", lambda: "spawn")
    return calls


def test_default_workers_fall_back_to_zero_with_a_warning(reruns, caplog):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert (
            main_guard.resolve_dataloader_workers(4, explicit=False, action="val()")
            == 0
        )
    (record,) = _guard_warnings(caplog)
    assert "val()" in record.getMessage()
    assert 'if __name__ == "__main__":' in record.getMessage()
    assert "workers=0" in record.getMessage()


def test_explicit_workers_are_kept_with_a_warning(reruns, caplog):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert (
            main_guard.resolve_dataloader_workers(6, explicit=True, action="val()") == 6
        )
    (record,) = _guard_warnings(caplog)
    assert "workers=6" in record.getMessage()


@pytest.mark.parametrize("explicit", [True, False])
def test_zero_workers_need_no_check(reruns, caplog, explicit):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert (
            main_guard.resolve_dataloader_workers(0, explicit=explicit, action="val()")
            == 0
        )
    assert reruns == []
    assert _guard_warnings(caplog) == []


def test_torchrun_ranks_are_left_alone(reruns, caplog, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert (
            main_guard.resolve_dataloader_workers(4, explicit=False, action="train()")
            == 4
        )
    assert reruns == []
    assert _guard_warnings(caplog) == []


def test_multi_device_ddp_launch_is_left_alone(reruns, caplog):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert (
            main_guard.resolve_dataloader_workers(
                4, explicit=False, action="train()", device="0,1"
            )
            == 4
        )
    assert reruns == []
    assert _guard_warnings(caplog) == []


def test_guarded_or_inconclusive_scripts_keep_their_workers(monkeypatch, caplog):
    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", lambda: False)
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert (
            main_guard.resolve_dataloader_workers(4, explicit=False, action="val()")
            == 4
        )
    assert _guard_warnings(caplog) == []


# ---------------------------------------------------------------------------
# train() wrapper
# ---------------------------------------------------------------------------


class _FakeModel:
    pass


def _kwargs_train(captured):
    def train(self, data, **kwargs):
        captured.update(kwargs)

    return train


def _named_train(captured, workers=4):
    def train(self, data, *, workers=workers, device="", **kwargs):
        captured.update(kwargs, workers=workers, device=device)

    return train


def test_train_default_workers_fall_back_to_zero(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_kwargs_train(captured))(_FakeModel(), "data.yaml")
    assert captured["workers"] == 0
    (record,) = _guard_warnings(caplog)
    assert "train()" in record.getMessage()


def test_train_named_default_workers_fall_back_to_zero(reruns):
    captured = {}
    _wrap_train_with_cfg(_named_train(captured))(_FakeModel(), "data.yaml")
    assert captured["workers"] == 0


def test_train_family_default_of_zero_needs_no_warning(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_named_train(captured, workers=0))(_FakeModel(), "d.yaml")
    assert captured["workers"] == 0
    assert _guard_warnings(caplog) == []


def test_train_explicit_workers_are_kept_with_a_warning(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_kwargs_train(captured))(_FakeModel(), "d.yaml", workers=3)
    assert captured["workers"] == 3
    (record,) = _guard_warnings(caplog)
    assert "workers=3" in record.getMessage()


def test_train_workers_from_cfg_count_as_explicit(reruns, tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("workers: 2\n", encoding="utf-8")
    captured = {}
    _wrap_train_with_cfg(_kwargs_train(captured))(_FakeModel(), "d.yaml", cfg=str(cfg))
    assert captured["workers"] == 2


def test_train_explicit_zero_workers_need_no_warning(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_kwargs_train(captured))(_FakeModel(), "d.yaml", workers=0)
    assert captured["workers"] == 0
    assert _guard_warnings(caplog) == []


def test_train_multi_device_ddp_launch_is_left_alone(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_kwargs_train(captured))(
            _FakeModel(), "d.yaml", device="0,1"
        )
    assert captured == {"device": "0,1"}
    assert _guard_warnings(caplog) == []


def test_train_under_torchrun_is_left_alone(reruns, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    captured = {}
    _wrap_train_with_cfg(_kwargs_train(captured))(_FakeModel(), "d.yaml", epochs=1)
    assert captured == {"epochs": 1}


def test_train_arguments_unchanged_when_not_detected(monkeypatch):
    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", lambda: False)
    captured = {}
    _wrap_train_with_cfg(_kwargs_train(captured))(_FakeModel(), "d.yaml", epochs=1)
    assert captured == {"epochs": 1}


def _positional_train(captured):
    def train(self, data, workers=4, device=""):
        captured.update(workers=workers, device=device)

    return train


def test_train_explicit_positional_workers_are_kept(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_positional_train(captured))(_FakeModel(), "d.yaml", 3)
    assert captured["workers"] == 3
    (record,) = _guard_warnings(caplog)
    assert "workers=3" in record.getMessage()


def test_train_positional_multi_device_is_left_alone(reruns, caplog):
    captured = {}
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        _wrap_train_with_cfg(_positional_train(captured))(
            _FakeModel(), "d.yaml", 4, "0,1"
        )
    assert captured == {"workers": 4, "device": "0,1"}
    assert reruns == []
    assert _guard_warnings(caplog) == []


def test_train_family_default_of_none_falls_back_to_zero(reruns):
    captured = {}
    _wrap_train_with_cfg(_named_train(captured, workers=None))(_FakeModel(), "d.yaml")
    assert captured["workers"] == 0


class _ResumingModel:
    """Restores ``workers=6`` from its checkpoint, like a resumed family run."""

    RESUME_RESTORES_TRAIN_ARGS = True

    def _restore_resume_interpolation(self, resume):
        pass

    def _checkpoint_train_config(self, resume):
        return {}

    def _resume_train_args(self, resume, given):
        return {} if "workers" in given else {"workers": 6}


def test_train_resume_restored_workers_are_not_explicit(reruns):
    captured = {}
    _wrap_train_with_cfg(_kwargs_train(captured))(
        _ResumingModel(), "d.yaml", resume="last.pt"
    )
    assert captured["workers"] == 0


def test_train_resume_keeps_workers_passed_with_it(reruns):
    captured = {}
    _wrap_train_with_cfg(_kwargs_train(captured))(
        _ResumingModel(), "d.yaml", resume="last.pt", workers=2
    )
    assert captured["workers"] == 2


def test_train_resume_restored_workers_kept_when_not_detected(monkeypatch):
    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", lambda: False)
    captured = {}
    _wrap_train_with_cfg(_kwargs_train(captured))(
        _ResumingModel(), "d.yaml", resume="last.pt"
    )
    assert captured["workers"] == 6


@pytest.fixture
def rfdetr_resume(tmp_path, monkeypatch):
    """Resume an RF-DETR ``last.pt`` saved with ``workers=4``; return trainer kwargs."""
    import torch

    from libreyolo.models.rfdetr import model as rfdetr_model
    from libreyolo.models.rfdetr.trainer import RFDETRConfig

    last = tmp_path / "runs" / "rw" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    rfdetr_model.LibreRFDETR({}, size="n", nb_classes=2, device="cpu").save(str(last))
    config = RFDETRConfig(
        data="data.yaml", workers=4, project=str(tmp_path / "runs"), name="rw"
    ).to_dict()
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    torch.save({**checkpoint, "epoch": 0, "config": config}, last)

    captured = {}

    class _Trainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured.update(kwargs)

        def setup(self):
            pass

        def resume(self, checkpoint_path):
            pass

        def train(self):
            return {}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _Trainer)
    model = rfdetr_model.LibreRFDETR(None, size="n", nb_classes=2, device="cpu")

    def run(**kwargs):
        model.train(resume=str(last), **kwargs)
        return captured

    return run


def test_rfdetr_resume_restored_workers_fall_back_to_zero(reruns, rfdetr_resume):
    assert rfdetr_resume()["workers"] == 0


def test_rfdetr_resume_keeps_workers_passed_with_it(reruns, rfdetr_resume):
    assert rfdetr_resume(workers=2)["workers"] == 2


def test_rfdetr_resume_restored_workers_kept_when_not_detected(
    monkeypatch, rfdetr_resume
):
    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", lambda: False)
    assert rfdetr_resume()["workers"] == 4


@pytest.fixture
def dinov2_resume(tmp_path, monkeypatch):
    """Resume a DINOv2 ``last.pt`` saved with ``workers=4``; return trainer kwargs."""
    import torch
    from torch import nn

    import libreyolo.models.dinov2.trainer as dinov2_trainer
    import libreyolo.models.rfdetr.nn as rfdetr_nn
    from libreyolo.models.dinov2.config import DINOv2Config
    from libreyolo.models.dinov2.model import LibreDINOv2

    # No pretrained backbone download; the stub trainer never runs it.
    monkeypatch.setattr(
        rfdetr_nn,
        "build_backbone",
        lambda load_dinov2_weights=True, **kwargs: nn.Sequential(nn.Identity()),
    )
    last = tmp_path / "runs" / "dw" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    config = DINOv2Config(data="saved.yaml", workers=4).to_dict()
    torch.save({"epoch": 0, "config": config}, last)

    captured = {}

    class _Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def setup(self):
            pass

        def resume(self, path):
            pass

        def train(self):
            return {}

    monkeypatch.setattr(dinov2_trainer, "DINOv2Trainer", _Trainer)
    model = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    def run(**kwargs):
        model.train(resume=str(last), **kwargs)
        return captured

    return run


def test_dinov2_resume_restored_workers_fall_back_to_zero(reruns, dinov2_resume):
    assert dinov2_resume()["workers"] == 0


def test_dinov2_resume_keeps_workers_passed_with_it(reruns, dinov2_resume):
    assert dinov2_resume(workers=2)["workers"] == 2


def test_dinov2_resume_restored_workers_kept_when_not_detected(
    monkeypatch, dinov2_resume
):
    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", lambda: False)
    assert dinov2_resume()["workers"] == 4


@pytest.mark.parametrize("fixture", ["rfdetr_resume", "dinov2_resume"])
def test_resume_restored_workers_kept_under_torchrun(
    reruns, request, monkeypatch, fixture
):
    """The resolver's torchrun check covers a restored count too.

    Setting ``LOCAL_RANK`` only makes the resolver see a torchrun rank (whose
    workers do not re-import the user's script); no distributed training runs.
    """
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert request.getfixturevalue(fixture)()["workers"] == 4


# ---------------------------------------------------------------------------
# BaseModel.val()
# ---------------------------------------------------------------------------


class _ConfigBuilt(Exception):
    pass


@pytest.fixture
def val_workers(monkeypatch):
    """Call ``val()`` up to its ValidationConfig and return ``num_workers``."""
    from libreyolo import LibreYOLO9, validation

    def capture(**kwargs):
        raise _ConfigBuilt(kwargs["num_workers"])

    monkeypatch.setattr(validation, "ValidationConfig", capture)
    model = LibreYOLO9(None, size="t", device="cpu")

    def run(**kwargs):
        with pytest.raises(_ConfigBuilt) as built:
            model.val(data="data.yaml", **kwargs)
        return built.value.args[0]

    return run


def test_val_default_workers_fall_back_to_zero(reruns, val_workers, caplog):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert val_workers() == 0
    (record,) = _guard_warnings(caplog)
    assert "val()" in record.getMessage()


def test_val_explicit_workers_are_kept_with_a_warning(reruns, val_workers, caplog):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert val_workers(workers=4) == 4
    (record,) = _guard_warnings(caplog)
    assert "workers=4" in record.getMessage()


def test_val_explicit_zero_workers_need_no_warning(reruns, val_workers, caplog):
    with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
        assert val_workers(workers=0) == 0
    assert _guard_warnings(caplog) == []


def test_val_default_is_unchanged_when_not_detected(monkeypatch, val_workers):
    monkeypatch.setattr(main_guard, "main_script_reruns_in_workers", lambda: False)
    workers = val_workers()
    assert workers == 4
    assert type(workers) is int


def test_val_signature_default_is_still_four():
    import inspect

    from libreyolo.models.base.model import BaseModel

    assert inspect.signature(BaseModel.val).parameters["workers"].default == 4


def test_rfdetr_val_override_still_forwards_its_zero_default(monkeypatch):
    import inspect

    from libreyolo.models.rfdetr.model import LibreRFDETR

    assert inspect.signature(LibreRFDETR.val).parameters["workers"].default == 0


# ---------------------------------------------------------------------------
# End to end: a real unguarded script under the spawn start method
# ---------------------------------------------------------------------------


def test_unguarded_spawn_script_trains_and_validates(tmp_path):
    from PIL import Image

    dataset = tmp_path / "ds"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (64, 64), (60 * i, 90, 120)).save(
                dataset / "images" / split / f"{i}.jpg"
            )
            (dataset / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = dataset / "data.yaml"
    data.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: thing\n",
        encoding="utf-8",
    )
    script = tmp_path / "unguarded.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import multiprocessing
            multiprocessing.set_start_method("spawn", force=True)
            from libreyolo import LibreYOLO9

            model = LibreYOLO9(None, size="t", device="cpu")
            model.val(data={str(data)!r}, imgsz=64, batch=2, plots=False)
            model.train(
                data={str(data)!r}, epochs=1, imgsz=64, batch=2,
                project={str(tmp_path / "runs")!r}, exist_ok=True,
            )
            print("SCRIPT_DONE")
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    env.pop("LOCAL_RANK", None)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "SCRIPT_DONE" in result.stdout
    assert "bootstrapping phase" not in output
    assert "val() runs at the top level" in output
    assert "train() runs at the top level" in output
