import io


def test_api_returns_prediction(app, client, predictor, make_image):
    app.config["MODEL_VERSION"] = "test-model"

    response = client.post(
        "/api/v1/predict",
        data={"file": (io.BytesIO(make_image()), "sample.png")},
    )

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {
        "label": "NORMAL",
        "score": 0.25,
        "model_version": "test-model",
    }
    assert predictor.calls == 1


def test_api_rejects_missing_file(client, predictor):
    response = client.post("/api/v1/predict")

    assert response.status_code == 400
    assert response.get_json() == {
        "error": {
            "code": "invalid_image",
            "message": "Please select an image file.",
        }
    }
    assert predictor.calls == 0


def test_api_rejects_empty_filename(client, predictor):
    response = client.post(
        "/api/v1/predict",
        data={"file": (io.BytesIO(b""), "")},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"
    assert predictor.calls == 0


def test_api_rejects_invalid_image(client, predictor):
    response = client.post(
        "/api/v1/predict",
        data={"file": (io.BytesIO(b"not an image"), "fake.png")},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"
    assert predictor.calls == 0


def test_api_rejects_unsupported_format(client, predictor, make_image):
    response = client.post(
        "/api/v1/predict",
        data={
            "file": (
                io.BytesIO(make_image(image_format="GIF")),
                "renamed.png",
            )
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"
    assert predictor.calls == 0


def test_api_rejects_large_dimensions(client, predictor, make_image):
    response = client.post(
        "/api/v1/predict",
        data={
            "file": (
                io.BytesIO(make_image(size=(101, 100))),
                "large.png",
            )
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"
    assert predictor.calls == 0


def test_api_rejects_oversized_request(app, client, predictor):
    app.config["MAX_CONTENT_LENGTH"] = 1024

    response = client.post(
        "/api/v1/predict",
        data={"file": (io.BytesIO(b"x" * 2048), "large.png")},
    )

    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "upload_too_large"
    assert predictor.calls == 0
