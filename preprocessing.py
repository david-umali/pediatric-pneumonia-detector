import io

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when an uploaded image fails validation."""


def preprocess_image(
    image_bytes: bytes,
    width: int,
    height: int,
    max_pixels: int = 16_000_000,
) -> np.ndarray:
    """Validate an image and return a normalized batch for inference."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise InvalidImageError("Please upload a PNG or JPEG image.")

            if source.width * source.height > max_pixels:
                raise InvalidImageError(
                    "Image dimensions are too large. Please use a smaller image."
                )

            img = source.convert("RGB")
            img = img.resize((width, height))

            img_array = np.array(img, dtype=np.float32)

    except Image.DecompressionBombError as exc:
        raise InvalidImageError(
            "Image dimensions are too large. Please use a smaller image."
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(
            "The file is not a readable image. Please upload a valid PNG or JPEG."
        ) from exc

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array
