"""Webcam lighting and noise. Needs numpy and cv2. All on S x S x 3 float32 BGR crops in [0, 1].

Two kinds, applied differently on purpose:

- ``Photometric`` (exposure, white balance, gamma, light falling across the face) is the room. It goes through the
  input and the target alike, and through both frames of a pair alike, so the delta changes the way a real delta would
  under that light.
- ``degrade`` (sensor noise, slight blur, JPEG) is the camera. It goes through the input only, and the target delta stays
  the clean one. The model thereby sees noise and learns to ignore it: a delta that copied or removed the noise would
  flicker and would wipe out skin texture.

There is no horizontal flip: a face and its hair parting are not symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import AugmentConfig

_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


@dataclass(frozen=True)
class Photometric:
    gain: float
    white_balance: tuple[float, float, float]
    gamma: float
    gradient_angle: float
    gradient: float

    @staticmethod
    def identity() -> Photometric:
        return Photometric(1.0, (1.0, 1.0, 1.0), 1.0, 0.0, 0.0)

    @staticmethod
    def sample(rng: np.random.Generator, config: AugmentConfig) -> Photometric:
        return Photometric(
            gain=float(2.0 ** rng.uniform(-config.exposure_stops, config.exposure_stops)),
            white_balance=tuple(float(g) for g in rng.uniform(1.0 - config.white_balance, 1.0 + config.white_balance, 3)),
            gamma=float(rng.uniform(1.0 - config.gamma, 1.0 + config.gamma)),
            gradient_angle=float(rng.uniform(0.0, 2.0 * np.pi)),
            gradient=float(rng.uniform(0.0, config.lighting_gradient)),
        )

    def field(self, size: int) -> np.ndarray:
        """S x S x 1: multiplies the light, 1 at the crop's center."""
        ramp = (np.arange(size, dtype=np.float32) + 0.5) / size - 0.5
        x, y = np.meshgrid(ramp, ramp)
        return (1.0 + self.gradient * (x * np.cos(self.gradient_angle) + y * np.sin(self.gradient_angle)) * 2.0)[..., None]

    def apply(self, crop: np.ndarray, light_field: np.ndarray | None = None) -> np.ndarray:
        if light_field is None:
            light_field = self.field(crop.shape[0])
        lit = crop * self.gain * np.array(self.white_balance, dtype=np.float32) * light_field
        return np.power(np.clip(lit, 0.0, 1.0), self.gamma, dtype=np.float32)


def degrade(crop: np.ndarray, rng: np.random.Generator, config: AugmentConfig) -> np.ndarray:
    """A fresh draw per frame, as a sensor does."""
    image = crop
    if config.blur_sigma > 0.0 and rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.2, config.blur_sigma)))
    read = float(rng.uniform(0.0, config.read_noise))
    shot = float(rng.uniform(0.0, config.shot_noise))
    if read > 0.0 or shot > 0.0:
        # Webcam noise is mostly in the luminance after the camera's own processing.
        luma = np.tensordot(image, _LUMA_BGR, axes=([2], [0]))[..., None]
        sigma = np.sqrt(read * read + shot * shot * np.maximum(luma, 0.0))
        shape = image.shape
        common = rng.standard_normal((shape[0], shape[1], 1), dtype=np.float32)
        separate = rng.standard_normal(shape, dtype=np.float32)
        image = np.clip(image + sigma * (0.8 * common + 0.6 * separate), 0.0, 1.0).astype(np.float32)
    if config.jpeg_quality_min < 100:
        quality = int(rng.integers(config.jpeg_quality_min, 101))
        if quality < 100:
            ok, encoded = cv2.imencode(".jpg", (image * 255.0 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    return image.astype(np.float32)
