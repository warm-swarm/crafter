"""Texture randomisation for Crafter.

Implements a ``TextureBank`` that serves the same sprite-lookup interface as
``engine.Textures`` but applies a per-variant HSV perturbation to *world*
sprites while leaving *UI* sprites untouched. This will facilitate benchmarking
RL models for colour invariance and implementation of clour data augmentation
during training.
"""

import pathlib

import imageio.v3 as imageio
import numpy as np
from PIL import Image


WORLD_SPRITES = frozenset(
    {
        "zombie",
        "wood",
        "water",
        "tree",
        "table",
        "stone",
        "skeleton",
        "sand",
        "plant",
        "plant-young",
        "plant-ripe",
        "path",
        "log",
        "leaves",
        "lava",
        "iron",
        "grass",
        "furnace",
        "fence",
        "diamond",
        "cow",
        "coal",
        "arrow-up",
        "arrow-down",
        "arrow-left",
        "arrow-right",
        "player",
        "player-down",
        "player-up",
        "player-left",
        "player-right",
        "player-sleep",
        "sapling",
    }
)

UI_SPRITES = frozenset(
    {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "debug",
        "debug-2",
        "debug-3",
        "drink",
        "energy",
        "food",
        "health",
        "iron_pickaxe",
        "iron_sword",
        "stone_pickaxe",
        "stone_sword",
        "wood_pickaxe",
        "wood_sword",
        "unknown",
    }
)

# Identity is first in each list so that variant 0 = (0°, 1.0, 1.0).
_HUE_SHIFTS = tuple(range(0, 360, 30))  # 12 values, 30° apart
_SAT_MULTS = (1.0, 0.7, 1.3)  # identity first
_BRIGHT_MULTS = (1.0, 0.8, 1.2)  # identity first

NUM_HUES = len(_HUE_SHIFTS)
NUM_SATS = len(_SAT_MULTS)
NUM_BRIGHTS = len(_BRIGHT_MULTS)
NUM_VARIANTS = NUM_HUES * NUM_SATS * NUM_BRIGHTS  # 108


def variant_to_hsv(variant_id):
    """Pure mapping variant_id -> (hue_shift_deg, sat_mult, bright_mult)."""
    if not 0 <= variant_id < NUM_VARIANTS:
        raise ValueError(f"variant_id {variant_id} out of range [0, {NUM_VARIANTS}).")
    hue_idx, rest = divmod(variant_id, NUM_SATS * NUM_BRIGHTS)
    sat_idx, bright_idx = divmod(rest, NUM_BRIGHTS)
    return _HUE_SHIFTS[hue_idx], _SAT_MULTS[sat_idx], _BRIGHT_MULTS[bright_idx]


def _build_pools():
    train, test = [], []
    for v in range(NUM_VARIANTS):
        hue = variant_to_hsv(v)[0]
        (train if hue < 180 else test).append(v)
    return frozenset(train), frozenset(test)


_TRAIN_POOL, _TEST_POOL = _build_pools()


def _rgb_to_hsv(rgb):
    """Vectorised RGB->HSV. Input/output in [0, 1], shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    safe_max = np.where(maxc > 0, maxc, 1.0)
    s = np.where(maxc > 0, delta / safe_max, 0.0)
    safe_delta = np.where(delta > 0, delta, 1.0)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.zeros_like(maxc)
    mask_r = maxc == r
    mask_g = (maxc == g) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    h = np.where(mask_r, bc - gc, h)
    h = np.where(mask_g, 2.0 + rc - bc, h)
    h = np.where(mask_b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _hsv_perturb(image, hue_shift_deg, sat_mult, bright_mult):
    """Apply HSV perturbation to an RGB or RGBA uint8 image."""
    has_alpha = image.shape[-1] == 4
    if has_alpha:
        rgb = image[..., :3]
        alpha = image[..., 3:]
    else:
        rgb = image
        alpha = None
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = _rgb_to_hsv(rgb_f)
    hsv[..., 0] = (hsv[..., 0] + hue_shift_deg / 360.0) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * bright_mult, 0.0, 1.0)
    rgb_out = _hsv_to_rgb(hsv)
    rgb_u8 = np.clip(np.rint(rgb_out * 255.0), 0, 255).astype(np.uint8)
    if has_alpha:
        return np.concatenate([rgb_u8, alpha], axis=-1)
    return rgb_u8


class TextureBank:
    """Drop-in replacement for ``engine.Textures`` with a per-variant perturbation.

    Variant 0 is the identity transform: the pipeline is skipped and sprites
    match ``engine.Textures`` byte-for-byte. For other variants, a deterministic
    HSV shift is applied to every *world* sprite; UI sprites pass through
    unchanged.
    """

    TRAIN_POOL = _TRAIN_POOL
    TEST_POOL = _TEST_POOL
    NUM_VARIANTS = NUM_VARIANTS
    WORLD_SPRITES = WORLD_SPRITES
    UI_SPRITES = UI_SPRITES

    def __init__(self, directory, variant_id=0):
        self._directory = pathlib.Path(directory)
        self._validate_assets()
        self._originals_by_variant = {}
        self._textures = {}
        self._variant_id = None
        self.set_variant(variant_id)

    def _validate_assets(self):
        found = {p.stem for p in self._directory.glob("*.png")}
        expected = WORLD_SPRITES | UI_SPRITES
        unexpected = found - expected
        if unexpected:
            raise ValueError(
                f"TextureBank: unexpected PNG file(s) in {self._directory}: "
                f"{sorted(unexpected)}. Each asset must be classified as either a "
                "world sprite (perturbed) or a UI sprite (passthrough). Update "
                "crafter.textures.WORLD_SPRITES / UI_SPRITES to classify it."
            )
        missing = expected - found
        if missing:
            raise ValueError(
                f"TextureBank: expected PNG file(s) missing from {self._directory}: "
                f"{sorted(missing)}."
            )

    def _build_bank(self, variant_id):
        hue, sat, bright = variant_to_hsv(variant_id)
        bank = {}
        for filename in self._directory.glob("*.png"):
            image = imageio.imread(filename.read_bytes())
            image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
            name = filename.stem
            if variant_id != 0 and name in WORLD_SPRITES:
                image = _hsv_perturb(image, hue, sat, bright)
            bank[name] = image
            self._textures[(variant_id, name, image.shape[:2])] = image
        return bank

    def set_variant(self, variant_id):
        variant_id = int(variant_id)
        if not 0 <= variant_id < NUM_VARIANTS:
            raise ValueError(
                f"variant_id {variant_id} out of range [0, {NUM_VARIANTS})."
            )
        if variant_id not in self._originals_by_variant:
            self._originals_by_variant[variant_id] = self._build_bank(variant_id)
        self._variant_id = variant_id

    @property
    def variant_id(self):
        return self._variant_id

    def get(self, name, size):
        if name is None:
            name = "unknown"
        size = int(size[0]), int(size[1])
        key = (self._variant_id, name, size)
        if key not in self._textures:
            image = self._originals_by_variant[self._variant_id][name]
            pil = Image.fromarray(image)
            pil = pil.resize(size[::-1], resample=Image.NEAREST)
            self._textures[key] = np.array(pil)
        return self._textures[key]
