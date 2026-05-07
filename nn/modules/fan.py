from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


def build_box_prior(bboxes, batch_idx, batch_size, image_size, device=None, dtype=torch.float32):
    """Build a dense object prior map from normalized xywh labels."""
    if device is None:
        device = bboxes.device if bboxes is not None else batch_idx.device

    h, w = image_size
    prior = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)
    if bboxes is None or bboxes.numel() == 0:
        return prior

    boxes = bboxes.detach().to(device=device, dtype=torch.float32)
    boxes_xyxy = torch.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    boxes_xyxy[:, [0, 2]] *= w
    boxes_xyxy[:, [1, 3]] *= h

    x1 = torch.floor(boxes_xyxy[:, 0]).clamp(0, w - 1).long()
    y1 = torch.floor(boxes_xyxy[:, 1]).clamp(0, h - 1).long()
    x2 = torch.ceil(boxes_xyxy[:, 2]).clamp(0, w).long()
    y2 = torch.ceil(boxes_xyxy[:, 3]).clamp(0, h).long()
    valid = (x2 > x1) & (y2 > y1)
    if not valid.any():
        return prior

    batch_ids = batch_idx.to(device=device, dtype=torch.long).view(-1)[valid]
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]

    width = w + 1
    area = (h + 1) * width
    base = batch_ids * area
    diff = torch.zeros(batch_size * area, device=device, dtype=dtype)
    ones = torch.ones_like(x1, device=device, dtype=dtype)

    idx_tl = base + y1 * width + x1
    idx_tr = base + y1 * width + x2
    idx_bl = base + y2 * width + x1
    idx_br = base + y2 * width + x2

    diff.scatter_add_(0, idx_tl, ones)
    diff.scatter_add_(0, idx_tr, -ones)
    diff.scatter_add_(0, idx_bl, -ones)
    diff.scatter_add_(0, idx_br, ones)

    prior = diff.view(batch_size, h + 1, w + 1).cumsum(1).cumsum(2)[:, :h, :w]
    return prior.clamp(0.0, 1.0).unsqueeze(1)


def resize_prior(prior, size, mode="nearest", preserve_small=False):
    """Resize a prior map to the target spatial size."""
    if prior is None:
        return None
    if prior.shape[-2:] == size:
        return prior
    if preserve_small:
        src_h, src_w = prior.shape[-2:]
        dst_h, dst_w = size
        if src_h >= dst_h and src_w >= dst_w:
            # Preserve tiny-object supervision when shrinking GT box priors to P3/P4 maps.
            pooled = F.adaptive_max_pool2d(prior.float(), size)
            return pooled.to(dtype=prior.dtype)
    if mode == "nearest":
        return F.interpolate(prior, size=size, mode=mode)
    return F.interpolate(prior, size=size, mode=mode, align_corners=False)


def make_edge_mask(mask, kernel_size=3):
    """Approximate a boundary strip with dilation minus erosion."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def split_prior_masks(prior, edge_kernel=3):
    """Split a prior map into object, edge, and background masks."""
    obj = prior.clamp(0.0, 1.0)
    edge = make_edge_mask(obj, edge_kernel)
    bg = 1.0 - obj
    return obj, edge, bg


def split_guidance_masks(prior, edge_kernel=3, neutral_band=0.15):
    """Build confidence-aware object/edge/background masks for forward guidance."""
    prior = prior.clamp(0.0, 1.0)
    neutral_band = float(max(min(neutral_band, 0.49), 0.0))
    high = 0.5 + neutral_band
    low = 0.5 - neutral_band
    obj = ((prior - high) / max(1.0 - high, 1e-6)).clamp(0.0, 1.0)
    bg = ((low - prior) / max(low, 1e-6)).clamp(0.0, 1.0)
    confidence = torch.maximum(obj, bg)
    edge = make_edge_mask(prior, edge_kernel) * confidence
    return obj.clamp(0.0, 1.0), edge.clamp(0.0, 1.0), bg.clamp(0.0, 1.0)


def build_rfft_band_masks(h, w, cutoff, device, dtype):
    """Build high/low frequency masks for rfft2 outputs."""
    fy = torch.fft.fftfreq(h, device=device).view(h, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, w // 2 + 1)
    radius = torch.sqrt(fy.square() + fx.square())
    high = (radius >= cutoff).to(dtype=dtype).view(1, 1, h, w // 2 + 1)
    low = 1.0 - high
    return high, low


def masked_rfft(x, mask=None):
    """Compute amplitude and phase with an optional spatial mask."""
    x = x.float()
    if mask is not None:
        x = x * mask.float()
    spec = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    return spec.abs(), torch.angle(spec)


def wrapped_phase_l1(pred, target):
    """Phase distance that respects angular wraparound."""
    delta = pred - target
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs().mean()


class PriorPredictor(nn.Module):
    """Lightweight prior predictor used when GT boxes are unavailable."""

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or max(channels // 4, 16)
        self.net = nn.Sequential(
            Conv(channels, hidden, 3, 1),
            Conv(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class FixedHighPass(nn.Module):
    """Depthwise Laplacian high-pass operator used to emphasize local edges."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def forward(self, x):
        weight = self.kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(x, weight, stride=1, padding=1, groups=x.shape[1])


class RFSEB(nn.Module):
    """Residual Fourier spectral enhancement block with explicit amplitude/phase branches."""

    def __init__(self, channels, cutoff=0.25, edge_kernel=3):
        super().__init__()
        self.cutoff = cutoff
        self.edge_kernel = edge_kernel
        self.freq_channels = max(channels // 2, 16)
        self.high_gain = nn.Parameter(torch.tensor(0.5))
        self.low_gain = nn.Parameter(torch.tensor(0.5))
        self.phase_gain = nn.Parameter(torch.tensor(0.25))
        self.spatial_res_gain = nn.Parameter(torch.tensor(0.25))
        self.spatial_obj_gain = nn.Parameter(torch.tensor(0.35))
        self.spatial_edge_gain = nn.Parameter(torch.tensor(0.45))
        self.spatial_bg_gain = nn.Parameter(torch.tensor(0.25))
        self.reduce = Conv(channels, self.freq_channels, 1, 1, act=False)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(self.freq_channels * 2, self.freq_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.freq_channels),
        )
        self.phase_proj = nn.Sequential(
            nn.Conv2d(self.freq_channels * 2, self.freq_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.freq_channels),
        )
        self.spatial_highpass = FixedHighPass()
        self.reproj = Conv(self.freq_channels, channels, 1, 1, act=False)

    def forward(self, x, prior, return_cache=False):
        forward_prior = resize_prior(prior, x.shape[-2:], preserve_small=True).to(dtype=x.dtype)
        obj, edge, bg = split_guidance_masks(forward_prior, self.edge_kernel)
        freq_x = self.reduce(x)
        amp, phase = masked_rfft(freq_x)

        high_mask, low_mask = build_rfft_band_masks(freq_x.shape[-2], freq_x.shape[-1], self.cutoff, freq_x.device,
                                                    amp.dtype)
        obj_f = resize_prior(obj, amp.shape[-2:], mode="bilinear").to(dtype=amp.dtype)
        edge_f = resize_prior(edge, amp.shape[-2:], mode="bilinear").to(dtype=amp.dtype)
        bg_f = resize_prior(bg, amp.shape[-2:], mode="bilinear").to(dtype=amp.dtype)
        obj_s = resize_prior(obj, freq_x.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=freq_x.dtype)
        edge_s = resize_prior(edge, freq_x.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=freq_x.dtype)
        bg_s = resize_prior(bg, freq_x.shape[-2:], mode="bilinear", preserve_small=True).to(dtype=freq_x.dtype)

        amp_high = amp * high_mask
        amp_low = amp * low_mask
        alpha = torch.sigmoid(self.high_gain)
        beta = torch.sigmoid(self.low_gain)
        gamma = torch.sigmoid(self.phase_gain)
        amp_high_en = amp_high * (1.0 + alpha * obj_f)
        amp_low_en = amp_low * (1.0 - beta * bg_f).clamp_min(0.0)
        amp_pair = torch.cat((amp_high_en, amp_low_en), dim=1).to(dtype=x.dtype)
        amp_en = F.softplus(self.amp_fuse(amp_pair)).float()

        phase_pair = torch.cat((phase, phase * edge_f), dim=1).to(dtype=x.dtype)
        phase_delta = self.phase_proj(phase_pair).float()
        phase_en = phase + gamma.float() * phase_delta

        spec_en = torch.polar(amp_en.clamp_min(1e-6), phase_en)
        feat_en = torch.fft.irfft2(spec_en, s=freq_x.shape[-2:], dim=(-2, -1), norm="ortho").to(dtype=x.dtype)
        spatial_hf = torch.tanh(self.spatial_highpass(freq_x))
        spatial_gate = 1.0
        spatial_gate = spatial_gate + torch.sigmoid(self.spatial_obj_gain) * obj_s
        spatial_gate = spatial_gate + torch.sigmoid(self.spatial_edge_gain) * edge_s
        spatial_gate = spatial_gate - torch.sigmoid(self.spatial_bg_gain) * bg_s
        spatial_gate = spatial_gate.clamp(0.0, 2.0)
        spatial_residual = (
                torch.sigmoid(self.spatial_res_gain).to(dtype=feat_en.dtype)
                * spatial_hf.to(dtype=feat_en.dtype)
                * spatial_gate.to(dtype=feat_en.dtype)
        )
        feat_en = feat_en + spatial_residual
        out = x + self.reproj(feat_en)
        if not return_cache:
            return out

        return out, {
            "freq_raw": freq_x,
            "freq_enhanced": feat_en,
            "spatial_hf": spatial_hf,
            "spatial_gate": spatial_gate,
            "spatial_residual": spatial_residual,
            "obj_mask": obj,
            "edge_mask": edge,
            "bg_mask": bg,
            "cutoff": self.cutoff,
        }


class DirectionalDWConv(nn.Module):
    """Masked depthwise convolution used for fast directional aggregation."""

    def __init__(self, channels, mask):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("kernel_mask", mask.view(1, 1, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        weight = self.weight * self.kernel_mask
        return F.conv2d(x, weight, self.bias, stride=1, padding=1, groups=x.shape[1])


class EightDirectionSS2D(nn.Module):
    """CUDA-friendly 8-direction aggregation core."""

    def __init__(self, channels):
        super().__init__()
        self.value_proj = Conv(channels, channels, 1, 1, act=False)
        self.delta_proj = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.gate_proj = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.out_proj = Conv(channels * 8, channels, 1, 1, act=False)
        self.gate_mix = nn.Parameter(torch.tensor(0.25))
        masks = [
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ]
        self.branches = nn.ModuleList([DirectionalDWConv(channels, mask) for mask in masks])

    def forward(self, x, modulation=None, return_cache=False):
        value = self.value_proj(x)
        delta_orig = torch.sigmoid(self.delta_proj(value))
        gate_orig = torch.sigmoid(self.gate_proj(value))
        delta = delta_orig if modulation is None else (delta_orig * modulation).clamp(0.0, 1.0)
        if modulation is None:
            gate = gate_orig
        else:
            gate_scale = 1.0 + torch.sigmoid(self.gate_mix) * (modulation - 1.0)
            gate = (gate_orig * gate_scale).clamp(0.0, 1.0)
        directions = [branch(value) for branch in self.branches]
        context = self.out_proj(torch.cat(directions, dim=1)) * gate * delta
        if not return_cache:
            return context
        return context, {
            "delta_raw": delta_orig,
            "delta_map": delta,
            "gate_raw": gate_orig,
            "gate_map": gate,
            "modulation_map": modulation,
        }


class PatchSS2D(nn.Module):
    """Patch-level 8-direction selective scan."""

    def __init__(self, channels, patch_size=4, edge_kernel=3):
        super().__init__()
        self.patch_size = patch_size
        self.edge_kernel = edge_kernel
        self.obj_gain = nn.Parameter(torch.tensor(0.35))
        self.edge_gain = nn.Parameter(torch.tensor(0.20))
        self.bg_gain = nn.Parameter(torch.tensor(0.35))
        self.pre = Conv(channels, channels, 1, 1)
        self.scan = EightDirectionSS2D(channels)
        self.out = Conv(channels, channels, 1, 1, act=False)

    def _patchify(self, x):
        b, c, h, w = x.shape
        ps = self.patch_size
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2] // ps, x.shape[-1] // ps
        patches = x.view(b, c, hp, ps, wp, ps).permute(0, 1, 2, 4, 3, 5)
        patches = patches.reshape(b, c, hp, wp, ps * ps)
        # Preserve sparse tiny-object activations better than plain mean pooling.
        tokens = 0.5 * (patches.mean(-1) + patches.amax(-1))
        return tokens, (h, w, pad_h, pad_w)

    def _unpatchify(self, tokens, meta):
        h, w, _, _ = meta
        ps = self.patch_size
        b, c, hp, wp = tokens.shape
        x = tokens.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, ps, ps)
        x = x.reshape(b, c, hp, wp, ps, ps).permute(0, 1, 2, 4, 3, 5).reshape(b, c, hp * ps, wp * ps)
        return x[:, :, :h, :w]

    def forward(self, x, prior=None, return_cache=False):
        tokens, meta = self._patchify(self.pre(x))
        modulation = None
        obj_tokens = edge_tokens = bg_tokens = None
        if prior is not None:
            guide = resize_prior(prior, x.shape[-2:], preserve_small=True).to(dtype=tokens.dtype)
            obj, edge, bg = split_guidance_masks(guide, self.edge_kernel)
            obj_tokens, _ = self._patchify(obj)
            edge_tokens, _ = self._patchify(edge)
            bg_tokens, _ = self._patchify(bg)
            modulation = 1.0
            modulation = modulation + torch.sigmoid(self.obj_gain) * obj_tokens.to(dtype=tokens.dtype)
            modulation = modulation + torch.sigmoid(self.edge_gain) * edge_tokens.to(dtype=tokens.dtype)
            modulation = modulation - torch.sigmoid(self.bg_gain) * bg_tokens.to(dtype=tokens.dtype)
            modulation = modulation.clamp(0.0, 2.0)
        context = (
            self.scan(tokens, modulation=modulation)
            if not return_cache
            else self.scan(tokens, modulation=modulation, return_cache=True)
        )
        patch_cache = None
        if return_cache:
            context, scan_cache = context
            patch_cache = {
                "patch_obj_mask": obj_tokens,
                "patch_edge_mask": edge_tokens,
                "patch_bg_mask": bg_tokens,
                "patch_gate_map": scan_cache["gate_map"],
                "patch_delta_raw": scan_cache["delta_raw"],
                "patch_delta_map": scan_cache["delta_map"],
                "patch_modulation_map": scan_cache["modulation_map"],
            }
        context = self._unpatchify(context, meta)
        out = x + self.out(context)
        if not return_cache:
            return out
        return out, patch_cache


class BBoxGuidedSS2D(nn.Module):
    """8-direction selective scan modulated by box priors."""

    def __init__(self, channels, edge_kernel=3):
        super().__init__()
        self.edge_kernel = edge_kernel
        self.obj_gain = nn.Parameter(torch.tensor(0.5))
        self.edge_gain = nn.Parameter(torch.tensor(0.25))
        self.bg_gain = nn.Parameter(torch.tensor(0.5))
        self.pre = Conv(channels, channels, 1, 1)
        self.scan = EightDirectionSS2D(channels)
        self.out = Conv(channels, channels, 1, 1, act=False)

    def forward(self, x, prior, return_cache=False):
        obj, edge, bg = split_guidance_masks(
            resize_prior(prior, x.shape[-2:], preserve_small=True).to(dtype=x.dtype), self.edge_kernel
        )
        modulation = 1.0
        modulation = modulation + torch.sigmoid(self.obj_gain) * obj
        modulation = modulation + torch.sigmoid(self.edge_gain) * edge
        modulation = modulation - torch.sigmoid(self.bg_gain) * bg
        modulation = modulation.clamp(0.0, 2.0)

        context = self.scan(self.pre(x), modulation=modulation) if not return_cache else self.scan(
            self.pre(x), modulation=modulation, return_cache=True
        )
        if not return_cache:
            return x + self.out(context)

        context, scan_cache = context
        out = x + self.out(context)
        return out, {
            "obj_mask": obj,
            "edge_mask": edge,
            "bg_mask": bg,
            "modulation_map": modulation,
            "delta_raw": scan_cache["delta_raw"],
            "delta_map": scan_cache["delta_map"],
            "gate_raw": scan_cache["gate_raw"],
            "gate_map": scan_cache["gate_map"],
        }


class FANBlock(nn.Module):
    """FAN block: RF-SEB -> Patch SS2D -> Box-guided SS2D."""

    def __init__(self, c1, c2, patch_size=4, cutoff=0.25, edge_kernel=3):
        super().__init__()
        self.stem = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.prior_predictor = PriorPredictor(c2)
        self.rfseb = RFSEB(c2, cutoff=cutoff, edge_kernel=edge_kernel)
        self.patch_scan = PatchSS2D(c2, patch_size=patch_size, edge_kernel=edge_kernel)
        self.box_scan = BBoxGuidedSS2D(c2, edge_kernel=edge_kernel)
        self.out = Conv(c2, c2, 1, 1, act=False)

    def forward(self, x, prior=None, return_cache=False, guide_mix=0.0):
        x = self.stem(x)
        prior_logits = self.prior_predictor(x)
        pred_prior = torch.sigmoid(prior_logits)
        supervision_prior = pred_prior.detach()
        if prior is not None:
            gt_prior = resize_prior(prior, x.shape[-2:], preserve_small=True).to(dtype=x.dtype)
            mix = float(max(min(guide_mix, 1.0), 0.0)) if self.training else 0.0
            guide = (1.0 - mix) * gt_prior + mix * pred_prior if self.training else gt_prior
            supervision_prior = gt_prior
        else:
            guide = pred_prior

        if return_cache:
            x, freq_cache = self.rfseb(x, guide, return_cache=True)
            x, patch_cache = self.patch_scan(x, guide, return_cache=True)
            x, scan_cache = self.box_scan(x, guide, return_cache=True)
        else:
            x = self.rfseb(x, guide, return_cache=False)
            x = self.patch_scan(x, guide)
            x = self.box_scan(x, guide, return_cache=False)
            return self.out(x)

        obj_mask, edge_mask, bg_mask = split_prior_masks(supervision_prior, self.box_scan.edge_kernel)
        out = self.out(x)
        cache = {
            **freq_cache,
            **(patch_cache or {}),
            "prior_logits": prior_logits,
            "guide_map": guide,
            "semantic_feat": out,
            "obj_mask": obj_mask,
            "edge_mask": edge_mask,
            "bg_mask": bg_mask,
            "modulation_map": scan_cache["modulation_map"],
            "delta_raw": scan_cache["delta_raw"],
            "delta_map": scan_cache["delta_map"],
            "gate_raw": scan_cache["gate_raw"],
            "gate_map": scan_cache["gate_map"],
        }
        return out, cache


__all__ = (
    "FANBlock",
    "PriorPredictor",
    "build_box_prior",
    "build_rfft_band_masks",
    "make_edge_mask",
    "masked_rfft",
    "resize_prior",
    "split_guidance_masks",
    "split_prior_masks",
    "wrapped_phase_l1",
)
