import numpy as np
import pytest

from pneumonia_detector.preprocessing import InvalidImageError, preprocess_image


def test_converts_grayscale_to_normalized_rgb_batch(make_image):
    image_bytes = make_image(
        size=(12, 8),
        mode="L",
        color=128,
    )

    result = preprocess_image(image_bytes, width=6, height=4)

    assert result.shape == (1, 4, 6, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 128 / 255.0, atol=1e-7)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_accepts_supported_formats(make_image, image_format):
    image_bytes = make_image(image_format=image_format)

    result = preprocess_image(image_bytes, width=8, height=8)

    assert result.shape == (1, 8, 8, 3)


def test_rejects_invalid_image_bytes():
    with pytest.raises(InvalidImageError, match="not a readable image"):
        preprocess_image(b"not an image", width=8, height=8)


def test_rejects_unsupported_format(make_image):
    image_bytes = make_image(image_format="GIF")

    with pytest.raises(InvalidImageError, match="PNG or JPEG"):
        preprocess_image(image_bytes, width=8, height=8)


def test_rejects_excessive_pixel_count(make_image):
    image_bytes = make_image(size=(11, 10))

    with pytest.raises(InvalidImageError, match="dimensions are too large"):
        preprocess_image(
            image_bytes,
            width=8,
            height=8,
            max_pixels=100,
        )


def test_accepts_image_at_pixel_limit(make_image):
    image_bytes = make_image(size=(10, 10))

    result = preprocess_image(
        image_bytes,
        width=8,
        height=8,
        max_pixels=100,
    )

    assert result.shape == (1, 8, 8, 3)
