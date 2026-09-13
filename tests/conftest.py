import io

import pytest
from PIL import Image

from app import create_app


class FakePredictor:
    input_width = 32
    input_height = 32

    def __init__(self):
        self.calls = 0

    def predict(self, img_array):
        self.calls += 1

        return {
            "label": "NORMAL",
            "score": 0.25,
        }


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def app(predictor):
    return create_app(
        config={
            "TESTING": True,
            "MODEL_PATH": "model/does-not-exist.h5",
            "MAX_CONTENT_LENGTH": 1024 * 1024,
            "MAX_IMAGE_PIXELS": 10_000,
        },
        predictor=predictor,
    )


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def make_image():
    def build(size=(20, 20), mode="RGB", color="white", image_format="PNG"):
        buffer = io.BytesIO()

        with Image.new(mode, size, color=color) as image:
            image.save(buffer, format=image_format)

        return buffer.getvalue()

    return build
