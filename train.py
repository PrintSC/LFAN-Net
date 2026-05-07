from __future__ import annotations

import csv
import os
import platform
import re
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import torch

from cli.parser import parse_args

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings(
    "ignore",
    message=r".*Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*",
    category=UserWarning,
)

AUTO_BATCH_AT_896 = {"n": 32, "s": 24, "m": 16, "l": 8, "x": 4}


class Tee:
    """Mirror stdout/stderr to both console and a log file."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class PlainTextLogStream:
    """Write a clean ASCII log file while preserving rich console output."""

    ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    REPLACEMENTS = {
        "\u26a0\ufe0f": "[WARN]",
        "\u26a0": "[WARN]",
        "\u2705": "[OK]",
        "\u274c": "[ERR]",
        "\U0001f680": "",
        "\u00b1": "+/-",
        "\u2192": "->",
        "\u2588": "#",
        "\u2589": "#",
        "\u258a": "#",
        "\u258b": "#",
        "\u258c": "#",
        "\u258d": "#",
        "\u258e": "#",
        "\u258f": "#",
    }

    def __init__(self, fp):
        self.fp = fp
        self.pending = ""
        self.encoding = "utf-8-sig"

    def _sanitize(self, data: str) -> str:
        text = self.ANSI_RE.sub("", data).replace("\r\n", "\n")
        for src, dst in self.REPLACEMENTS.items():
            text = text.replace(src, dst)
        return text.encode("ascii", "ignore").decode("ascii")

    def write(self, data):
        text = self._sanitize(data)
        for ch in text:
            if ch == "\r":
                self.pending = ""
            elif ch == "\n":
                self.fp.write(self.pending.rstrip() + "\n")
                self.pending = ""
            else:
                self.pending += ch
        return len(data)

    def flush(self):
        self.fp.flush()

    def finalize(self):
        if self.pending:
            self.fp.write(self.pending.rstrip() + "\n")
            self.pending = ""
        self.fp.flush()

    def isatty(self):
        return False


def max_imgsz(imgsz) -> int:
    if isinstance(imgsz, (list, tuple)):
        return max(int(x) for x in imgsz)
    return int(imgsz)


def primary_cuda_index(device) -> int | None:
    device = normalize_device(device)
    if not device:
        return None
    device = str(device).strip().lower()
    if device in {"cpu", "mps"}:
        return None
    head = device.split(",")[0].strip()
    return int(head) if head.isdigit() else 0


def get_cuda_memory_gb(device) -> float | None:
    if not torch.cuda.is_available():
        return None
    index = primary_cuda_index(device)
    if index is None:
        return None
    try:
        total_memory = torch.cuda.get_device_properties(index).total_memory
    except Exception:
        return None
    return total_memory / (1024 ** 3)


def extract_fan_scale(model_path: str | Path) -> str | None:
    match = re.search(r"fan_yolo11([nslmx])", Path(model_path).stem)
    return match.group(1) if match else None


def recommended_auto_scales(cuda_mem_gb: float | None, imgsz) -> tuple[str, ...]:
    image_size = max_imgsz(imgsz)
    if cuda_mem_gb is not None and cuda_mem_gb >= 40 and image_size <= 1024:
        return ("m", "s", "n", "l", "x")
    if cuda_mem_gb is not None and cuda_mem_gb >= 24:
        return ("s", "n", "m", "l", "x")
    return ("n", "s", "m", "l", "x")


def resolve_model_path(model_arg: str | None, imgsz, cuda_mem_gb: float | None) -> tuple[str, bool, str | None]:
    if model_arg and str(model_arg).strip().lower() not in {"auto", "best"}:
        return model_arg, False, None

    scales = recommended_auto_scales(cuda_mem_gb, imgsz)
    for scale in scales:
        if Path(f"yolo11{scale}.pt").exists():
            note = (
                f"auto-selected fan_yolo11{scale}.yaml "
                f"(recommended for {max_imgsz(imgsz)}px, cuda_mem={cuda_mem_gb:.1f}GB)"
                if cuda_mem_gb is not None
                else f"auto-selected fan_yolo11{scale}.yaml"
            )
            return str(Path(f"cfg/models/FAN-Net/fan_yolo11{scale}.yaml")), True, note

    return str(Path("cfg/models/FAN-Net/fan_yolo11n.yaml")), True, "fallback to fan_yolo11n.yaml"


def resolve_pretrained_arg(model_path: str | Path, pretrained_arg):
    if isinstance(pretrained_arg, (str, Path)):
        return str(pretrained_arg)
    if pretrained_arg is not True:
        return pretrained_arg

    scale = extract_fan_scale(model_path)
    if not scale:
        return pretrained_arg

    base_weights = Path(f"yolo11{scale}.pt")
    return str(base_weights) if base_weights.exists() else pretrained_arg


def recommend_batch(model_path: str | Path, imgsz, batch: int, auto_selected: bool) -> tuple[int, str | None]:
    if not auto_selected or batch != 32:
        return batch, None

    scale = extract_fan_scale(model_path)
    if scale not in AUTO_BATCH_AT_896:
        return batch, None

    image_size = max_imgsz(imgsz)
    base_batch = AUTO_BATCH_AT_896[scale]
    scaled_batch = max(4, int(round((base_batch * (896 / image_size) ** 2) / 4) * 4))
    scaled_batch = min(batch, scaled_batch)
    if scaled_batch == batch:
        return batch, None

    note = f"batch auto-adjusted from {batch} to {scaled_batch} for fan_yolo11{scale} at {image_size}px"
    return scaled_batch, note


def resolve_data_path(data_arg: str, data_root: str | None) -> str:
    candidates = [Path(data_arg)]
    if data_root:
        candidates.append(Path(data_root) / data_arg)
    candidates.append(Path("datasets") / data_arg)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return data_arg


def normalize_imgsz(imgsz):
    if isinstance(imgsz, (list, tuple)):
        return imgsz[0] if len(imgsz) == 1 else list(imgsz)
    return imgsz


def normalize_device(device):
    if isinstance(device, (list, tuple)):
        if len(device) == 1:
            return str(device[0])
        return ",".join(map(str, device))
    return device


def print_results_summary(save_dir):
    results_csv = Path(save_dir) / "results.csv"
    if not results_csv.exists():
        return

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    last = rows[-1]
    summary_keys = [
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    parts = []
    for key in summary_keys:
        if key in last and last[key] != "":
            parts.append(f"{key}={last[key]}")
    if parts:
        print("final_metrics: " + ", ".join(parts))
    print(f"results_csv: {results_csv}")


def main(yolo_cls=None):
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.txt"

    with log_file.open("w", encoding="utf-8-sig") as log_fp:
        plain_log = PlainTextLogStream(log_fp)
        tee_out = Tee(sys.stdout, plain_log)
        tee_err = Tee(sys.stderr, plain_log)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            args = parse_args()

            if yolo_cls is None:
                from models import YOLO as lfan_yolo

                yolo_cls = lfan_yolo

            cuda_mem_gb = get_cuda_memory_gb(args.device)
            model_path, auto_model_selected, auto_model_note = resolve_model_path(args.model, args.imgsz, cuda_mem_gb)
            data_path = resolve_data_path(args.data, args.data_path)
            workers = args.workers
            if os.name == "nt" and workers == 8:
                workers = 0

            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True

            train_kwargs = vars(args).copy()
            for key in ("config", "task", "mode", "data_path", "model"):
                train_kwargs.pop(key, None)

            train_kwargs["data"] = data_path
            train_kwargs["imgsz"] = normalize_imgsz(args.imgsz)
            train_kwargs["device"] = normalize_device(args.device)
            train_kwargs["pretrained"] = resolve_pretrained_arg(model_path, args.pretrained)
            train_kwargs["batch"], batch_note = recommend_batch(
                model_path, train_kwargs["imgsz"], int(args.batch), auto_model_selected
            )
            train_kwargs["workers"] = workers

            print("=" * 80)
            print("FAN-Net Train Launcher")
            print(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"cwd: {Path.cwd()}")
            print(f"python: {sys.executable}")
            print(f"platform: {platform.platform()}")
            print(f"log_file: {log_file.resolve()}")
            print(f"model: {model_path}")
            if auto_model_note:
                print(f"selection: {auto_model_note}")
            if Path(model_path).stem in {"fan_net", "fan_net_fast"}:
                print("note: legacy FAN yaml without yolo11[n/s/m/l/x] suffix will default to nano scale.")
                print("note: recommended scale-aware models: fan_yolo11s.yaml or fan_yolo11m.yaml")
            scale = extract_fan_scale(model_path)
            if scale and args.pretrained is True:
                base_weights = Path(f"yolo11{scale}.pt")
                if base_weights.exists():
                    print(f"pretrained: explicit scale-matched weights -> {base_weights}")
                else:
                    print(f"pretrained: missing local {base_weights} -> this FAN scale may train from scratch offline")
            elif isinstance(train_kwargs["pretrained"], str):
                print(f"pretrained: explicit weights -> {train_kwargs['pretrained']}")
            else:
                print(f"pretrained: {train_kwargs['pretrained']}")
            print(f"data: {data_path}")
            print(f"device: {train_kwargs['device']}")
            if cuda_mem_gb is not None:
                print(f"cuda_mem_gb: {cuda_mem_gb:.1f}")
            print(f"batch: {train_kwargs['batch']}")
            if batch_note:
                print(f"batch_policy: {batch_note}")
            print(f"workers: {train_kwargs['workers']}")
            print(f"project: {train_kwargs.get('project')}")
            print(f"name: {train_kwargs.get('name')}")
            print("=" * 80)

            try:
                model = yolo_cls(model_path)
                model.train(**train_kwargs)
                if getattr(model, "trainer", None) is not None:
                    print(f"save_dir: {model.trainer.save_dir}")
                    print_results_summary(model.trainer.save_dir)
                print(f"log saved to: {log_file.resolve()}")
            except Exception:
                print("training failed, traceback follows:")
                traceback.print_exc()
                print(f"log saved to: {log_file.resolve()}")
                raise
            finally:
                plain_log.finalize()


if __name__ == "__main__":
    main()
