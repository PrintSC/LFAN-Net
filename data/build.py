# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from cfg import IterableSimpleNamespace
from data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from utils import LOGGER, RANK, colorstr
from utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Create a sampler that repeats indefinitely.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the dataloader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration
    over a dataset without recreating the sampler.

    Attributes:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id: int):  # noqa
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _fan_class_name(names, cls_idx: int) -> str:
    """Resolve a readable class name for logging."""
    if isinstance(names, dict):
        return str(names.get(int(cls_idx), cls_idx))
    if isinstance(names, (list, tuple)) and 0 <= int(cls_idx) < len(names):
        return str(names[int(cls_idx)])
    return str(cls_idx)


def _fan_unique_classes(label: Dict[str, Any]) -> np.ndarray:
    """Return sorted unique class ids contained in one image label."""
    cls = label.get("cls", None)
    if cls is None:
        return np.empty(0, dtype=np.int64)
    cls = np.asarray(cls).reshape(-1)
    if cls.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(cls.astype(np.int64, copy=False))


def _fan_class_boxes(label: Dict[str, Any], nc: int) -> np.ndarray:
    """Return per-class box counts for one image label."""
    cls = label.get("cls", None)
    if cls is None:
        return np.zeros(nc, dtype=np.float64)
    cls = np.asarray(cls).reshape(-1)
    if cls.size == 0:
        return np.zeros(nc, dtype=np.float64)
    return np.bincount(cls.astype(np.int64, copy=False), minlength=nc).astype(np.float64, copy=False)


def _build_fan_rare_profile(dataset) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Build image-level rare sampling profile.

    The sampler stays image-based to preserve the existing YOLO data pipeline, while class rarity is measured from box
    counts instead of image counts. This matches dense tiny-object detection better than counting only image presence.
    """
    hyp = getattr(dataset, "hyp", None)
    strength = float(getattr(hyp, "fan_rare_sampling", 0.0) or 0.0)
    if strength <= 0 or not hasattr(dataset, "labels") or not getattr(dataset, "labels", None):
        return None, None
    if getattr(dataset, "rect", False):
        return None, {"disabled": "rect"}

    nc = int(getattr(dataset, "data", {}).get("nc", 0)) if isinstance(getattr(dataset, "data", None), dict) else 0
    image_classes = [_fan_unique_classes(label) for label in dataset.labels]
    if nc <= 0:
        max_cls = max((int(classes.max()) for classes in image_classes if classes.size), default=-1)
        nc = max_cls + 1
    if nc <= 0:
        return None, None

    counts = np.zeros(nc, dtype=np.float64)
    for label in dataset.labels:
        counts += _fan_class_boxes(label, nc)

    valid = counts > 0
    if valid.sum() < 2:
        return None, None

    rare_pow = max(float(getattr(hyp, "fan_rare_pow", 0.5) or 0.5), 0.0)
    rare_cap = max(float(getattr(hyp, "fan_rare_cap", 4.0) or 4.0), 1.0)
    max_count = counts[valid].max()
    rarity = np.ones(nc, dtype=np.float64)
    rarity[valid] = np.power(max_count / counts[valid], rare_pow)
    rarity = np.clip(rarity, 1.0, rare_cap)

    image_boosts = np.ones(len(image_classes), dtype=np.float64)
    for i, classes in enumerate(image_classes):
        if classes.size == 0:
            continue
        image_boost = float(rarity[classes].max())
        image_boosts[i] = image_boost

    if not np.isfinite(image_boosts).all() or image_boosts.sum() <= 0:
        return None, None

    weights = 1.0 + strength * (image_boosts - 1.0)
    weights /= max(weights.mean(), 1e-12)
    total_epochs = max(int(getattr(hyp, "epochs", 1) or 1), 1)
    warmup_epochs = max(int(round(float(getattr(hyp, "warmup_epochs", 0.0) or 0.0))), 1)
    close_mosaic = max(int(getattr(hyp, "close_mosaic", 0) or 0), 0)
    decay_start = total_epochs - close_mosaic if 0 < close_mosaic < total_epochs else None
    names = getattr(dataset, "data", {}).get("names", {}) if isinstance(getattr(dataset, "data", None), dict) else {}
    rare_order = np.argsort(counts[valid])
    rare_indices = np.flatnonzero(valid)[rare_order][: min(5, int(valid.sum()))]
    rare_preview = [
        f"{_fan_class_name(names, idx)}:{int(counts[idx])}box->x{rarity[idx]:.2f}" for idx in rare_indices.tolist()
    ]
    info = {
        "basis": "boxes",
        "strength": strength,
        "pow": rare_pow,
        "cap": rare_cap,
        "warmup_epochs": warmup_epochs,
        "decay_start": decay_start,
        "weights_min": float(weights.min()),
        "weights_mean": float(weights.mean()),
        "weights_max": float(weights.max()),
        "rare_preview": rare_preview,
    }
    return torch.as_tensor(image_boosts, dtype=torch.double), info


class _FANRareSamplerMixin:
    """Shared staged rare-sampling schedule for single-GPU and DDP samplers."""

    def _fan_init(self, dataset, image_boosts: torch.Tensor):
        hyp = getattr(dataset, "hyp", None)
        self.image_boosts = image_boosts.detach().cpu().to(dtype=torch.double)
        self.target_strength = float(getattr(hyp, "fan_rare_sampling", 0.0) or 0.0)
        self.total_epochs = max(int(getattr(hyp, "epochs", 1) or 1), 1)
        self.warmup_epochs = max(int(round(float(getattr(hyp, "warmup_epochs", 0.0) or 0.0))), 1)
        manual_decay_start = int(getattr(hyp, "fan_rare_decay_start", -1) or -1)
        close_mosaic = max(int(getattr(hyp, "close_mosaic", 0) or 0), 0)
        if 0 <= manual_decay_start < self.total_epochs:
            self.decay_start_epoch = manual_decay_start
        else:
            self.decay_start_epoch = self.total_epochs - close_mosaic if 0 < close_mosaic < self.total_epochs else None

    def _fan_strength(self, epoch: int) -> float:
        strength = self.target_strength
        if self.warmup_epochs > 0:
            strength *= min((epoch + 1) / self.warmup_epochs, 1.0)
        if self.decay_start_epoch is not None and epoch >= self.decay_start_epoch:
            tail = max(self.total_epochs - self.decay_start_epoch, 1)
            progress = (epoch - self.decay_start_epoch + 1) / tail
            strength *= max(0.0, 1.0 - progress)
        return max(strength, 0.0)

    def _fan_weights(self, epoch: int) -> torch.Tensor:
        weights = 1.0 + self._fan_strength(epoch) * (self.image_boosts - 1.0)
        weights /= max(float(weights.mean()), 1e-12)
        return weights.to(dtype=torch.double)


class _FANWeightedSampler(_FANRareSamplerMixin, torch.utils.data.WeightedRandomSampler):
    """Weighted sampler with warmup and tail decay for rare-class emphasis."""

    def __init__(self, dataset, image_boosts: torch.Tensor, generator=None):
        self._fan_init(dataset, image_boosts)
        super().__init__(weights=self._fan_weights(0), num_samples=len(dataset), replacement=True, generator=generator)

    def set_epoch(self, epoch: int):
        self.weights = self._fan_weights(int(epoch))


class _WeightedDistributedSampler(_FANRareSamplerMixin, distributed.DistributedSampler):
    """Distributed sampler that keeps rare-class sampling active under DDP."""

    def __init__(self, dataset, weights: torch.Tensor, **kwargs):
        super().__init__(dataset, shuffle=False, **kwargs)
        self._fan_init(dataset, weights)

    def __iter__(self) -> Iterator:
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + self.epoch)
        weights = self._fan_weights(int(self.epoch))
        indices = torch.multinomial(weights, self.total_size, replacement=True, generator=generator).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: Dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
):
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
):
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    sampler = None
    rare_profile, rare_info = _build_fan_rare_profile(dataset) if shuffle else (None, None)
    if rare_info and rare_info.get("disabled") == "rect":
        LOGGER.warning("FAN rare sampling is disabled because rect training requires ordered batches.")
    if rare_profile is not None:
        if rank == -1:
            sampler = _FANWeightedSampler(dataset, image_boosts=rare_profile, generator=generator)
        else:
            sampler = _WeightedDistributedSampler(dataset, weights=rare_profile, rank=rank)
        preview = ", ".join(rare_info["rare_preview"]) if rare_info["rare_preview"] else "n/a"
        decay = rare_info["decay_start"] if rare_info["decay_start"] is not None else "off"
        LOGGER.info(
            "FAN rare sampling enabled "
            f"(basis={rare_info['basis']}, strength={rare_info['strength']:.2f}, pow={rare_info['pow']:.2f}, "
            f"cap={rare_info['cap']:.2f}, warmup={rare_info['warmup_epochs']}, decay_start={decay}, "
            f"image_weight={rare_info['weights_min']:.2f}/{rare_info['weights_mean']:.2f}/{rare_info['weights_max']:.2f}, "
            f"rare={preview})"
        )
    elif rank != -1:
        sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )


def check_source(source):
    """
    Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Examples:
        Check a file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        Check a webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        source_lower = source.lower()
        is_file = source_lower.rpartition(".")[-1] in (IMG_FORMATS | VID_FORMATS)
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch: int = 1, vid_stride: int = 1, buffer: bool = False, channels: int = 3):
    """
    Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | Path | torch.Tensor | PIL.Image | np.ndarray, optional): The input source for inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
