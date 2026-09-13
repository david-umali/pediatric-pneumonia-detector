import numpy as np
from tensorflow.keras.models import load_model


class PredictionService:
    def __init__(self, model_path: str):
        try:
            self.model = load_model(model_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Unable to load model from '{model_path}': {exc}"
            ) from exc

        self.input_height = self.model.input_shape[1]
        self.input_width = self.model.input_shape[2]

    def predict(self, img_array: np.ndarray) -> dict:
        prediction = self.model.predict(img_array, verbose=0)
        score = float(prediction[0][0])

        label = "PNEUMONIA" if score >= 0.5 else "NORMAL"

        return {
            "label": label,
            "score": score,
        }
