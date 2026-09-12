import io

import numpy as np
from PIL import Image


def preprocess_image(image_bytes: bytes, width: int, height: int) -> np.ndarray:
    """Convert image bytes into a normalized batch for model inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((width, height))

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array
