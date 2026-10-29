# -*- coding: utf-8 -*-
r"""Elevation-guided super-resolution: a wrapper, never an edit.

`flair_sr_core.py` is lifted verbatim from the notebook that produced the
published runs and must stay byte-identical, or the guided/unguided comparison
stops being a comparison. So guidance is added around the base model here.

WHERE THE GUIDE ENTERS, AND WHY IT IS NOT THE INPUT
---------------------------------------------------
nDSM lives at the TARGET grid: 512 px at 0.20 m, co-registered with the aerial
image the model is trying to reconstruct. The coarse input is 10x10 (Sentinel-2)
or 64x64 (SPOT). Concatenating the guide to the input therefore means
downsampling it by 8x-51x first, which destroys the one thing it is carrying -
high-frequency structure the optical input physically cannot resolve. Building
edges and tree crowns survive at 0.20 m and do not survive at 10.24 m.

So the base model runs unchanged and produces its prediction at the target grid,
and the guide meets it there, at full resolution, in a refinement head.

WHY THE HEAD IS ZERO-INITIALISED
--------------------------------
The last convolution starts at zero, so at initialisation the guided model
returns the base model's output EXACTLY. Guided is then a strict superset of
unguided: it can represent the baseline and has to learn its way off it. If the
guided arm fails to beat the unguided one, that is a real result about elevation
priors rather than an artefact of a harder optimisation problem or a different
initialisation.

WHICH REGIMES THIS IS FOR
-------------------------
The aerial regimes (`spot_to_aerial`, `s2_to_aerial`), where the target is
512 px and the DSM is natively 512 px on the same grid. For `s2_to_spot` the
target is 64 px SPOT, so the guide would have to be downsampled 8x to meet it
and most of its value is gone before it arrives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedSR(nn.Module):
    """Base SR model + a full-resolution elevation refinement head.

    forward(lo, size, guide) -> (B, 3, H, W)

    `guide` is (B, 1, H, W) nDSM in metres/30 (see the pack card), already on
    the target grid. It is resampled here only if the caller's target size
    differs, which happens for the 64 px SPOT target.
    """

    def __init__(self, base_cls, guide_ch: int = 1, width: int = 32,
                 depth: int = 3):
        super().__init__()
        self.base = base_cls()
        layers, c_in = [], 3 + guide_ch
        for _ in range(depth - 1):
            layers += [nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(inplace=True)]
            c_in = width
        last = nn.Conv2d(c_in, 3, 3, padding=1)
        # zero-init: guided == unguided at step 0, so the comparison is fair
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.head = nn.Sequential(*layers)

    def forward(self, lo, size, guide):
        base = self.base(lo, size)
        if guide.shape[-2:] != base.shape[-2:]:
            guide = F.interpolate(guide, size=base.shape[-2:], mode="area")
        return (base + self.head(torch.cat([base, guide], dim=1))).clamp(0, 1)

    def base_parameters(self):
        return self.base.parameters()

    def head_parameters(self):
        return self.head.parameters()


def build(arch: str, core, guided: bool, guide_ch: int = 1):
    """Return (model, takes_guide). `core` is the flair_sr_core module."""
    if arch == "ddpm":
        if guided:
            raise SystemExit("guided diffusion is not implemented")
        return DdpmSR(core), False
    cls = core.MODELS[arch]
    if not guided:
        return cls(), False
    if arch == "bicubic":
        raise SystemExit("bicubic is the control; guiding it makes it a model")
    return GuidedSR(cls, guide_ch=guide_ch), True


class DdpmSR(nn.Module):
    """Conditional diffusion, wearing the same interface as the other models.

    The regression arms expose `forward(lo, size) -> prediction`; diffusion has
    a different objective (predict the noise) and a different inference path
    (ancestral sampling). Rather than special-case it through the trainer, it is
    adapted here: `forward` samples, and `loss` gives the trainer the objective
    to minimise. Anything that can call an EDSR can call this.

    COST. Sampling is `steps` sequential forward passes per batch, so a full
    1,332-tile evaluation at 512 px is hours while the same evaluation at the
    64 px SPOT target is under a minute. The trainer therefore evaluates this
    arm once, at the end, rather than every epoch.

    The published diffusion run scored 256 tiles against its own bicubic
    reference, which made it incomparable to every other arm. This one is
    evaluated on the full test set against the shared control.
    """

    def __init__(self, core, steps: int = 200):
        super().__init__()
        self.net = core.DdpmUNet()
        self._core = core
        self.steps = steps

    @property
    def expensive_eval(self) -> bool:
        return True

    def loss(self, lo, y):
        """Noise-prediction MSE at a random timestep, conditioned on bicubic."""
        core = self._core
        cond = F.interpolate(lo, size=y.shape[-2:], mode="bicubic",
                             align_corners=False).clamp(0, 1)
        t = torch.randint(0, core.T_STEPS, (len(y),), device=y.device)
        a = core._alph[t][:, None, None, None]
        noise = torch.randn_like(y)
        xt = a.sqrt() * y + (1 - a).sqrt() * noise
        eps = self.net(xt, cond, t.float() / core.T_STEPS)
        return F.mse_loss(eps, noise)

    def forward(self, lo, size, guide=None):
        cond = F.interpolate(lo, size=size, mode="bicubic",
                             align_corners=False).clamp(0, 1)
        return self._core.ddpm_sample(self.net, cond, self.steps)
