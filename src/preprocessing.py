"""Единое детерминированное преобразование фотографий и картин."""
import hashlib
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .io import atomic_write


def load_rgb(path):
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_image(image, size=512):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def canny_map(image, low=100, high=200):
    if not 0 <= low < high <= 255:
        raise ValueError("Invalid Canny thresholds")
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=False)
    return Image.fromarray(edges).convert("RGB")


def pixel_sha(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int("".join("1" if value else "0" for value in (pixels[:, 1:] > pixels[:, :-1]).ravel()), 2)


def save_png(path, image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    atomic_write(path, buffer.getvalue())
