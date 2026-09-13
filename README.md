# Pediatric Pneumonia Detector

A Python and Flask web application that uses a TensorFlow/Keras convolutional neural network (CNN) to classify pediatric chest X-ray images as **NORMAL** or **PNEUMONIA**.

This educational and portfolio project demonstrates Python image preprocessing, model inference, and web request handling. Planned improvements include continuous integration, containerization, and Kubernetes deployment.

## Features

- Browser-based image upload
- Styled prediction results and upload error messages
- TensorFlow/Keras model inference
- Separate Python modules for preprocessing and prediction
- Dynamic image resizing based on model input dimensions
- RGB conversion and float32 pixel normalization
- Binary classification using a 0.5 threshold
- Configurable model location through `MODEL_PATH`
- Model loading once per application process
- Descriptive startup errors when model loading fails

## Technology Stack

- Python 3.12
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML / CSS
- Git

## Project Structure

```text
pediatric-pneumonia-detector/
├── app.py
├── inference.py
├── preprocessing.py
├── model/
│   └── cnn_best_100.h5
├── templates/
│   └── index.html
├── tests/
├── requirements-dev.txt
├── requirements.txt
├── README.md
└── .gitignore
```

| File | Responsibility |
| --- | --- |
| `app.py` | Create and configure Flask, accept a prediction dependency, and handle web routes |
| `inference.py` | Load the model and return classification labels and scores |
| `preprocessing.py` | Decode, resize, and normalize uploaded images |
| `templates/index.html` | Display the upload form, results, and errors |
| `requirements.txt` | Declare Python dependencies |

## Installation

### Clone the repository

```bash
git clone https://github.com/david-umali/pediatric-pneumonia-detector.git
cd pediatric-pneumonia-detector
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it using the command for your shell.

**Bash / Zsh:**

```bash
source .venv/bin/activate
```

**PowerShell on Windows:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Provide the trained model

The trained model is private, excluded from Git using `.gitignore`, and **not included in this repository**. Supply a compatible model before running the application.

Place a compatible model at:

```text
model/cnn_best_100.h5
```

Alternatively, configure its location using `MODEL_PATH`.

## Model Configuration

By default, the application loads `model/cnn_best_100.h5` relative to `app.py`.

Set `MODEL_PATH` to use a different location.

**Bash / Zsh:**

```bash
MODEL_PATH="/path/to/model.h5" python app.py
```

**PowerShell:**

```powershell
$env:MODEL_PATH = "C:\path\to\model.h5"
python app.py
```

To clear the setting in PowerShell:

```powershell
Remove-Item Env:MODEL_PATH
```

A relative `MODEL_PATH` is resolved from the current working directory. Use an absolute path when running the application from another directory.

If the model cannot be loaded, the application stops during startup with an error identifying the configured path.

### Model compatibility

The current implementation assumes:

- A single image input with shape `(batch, height, width, 3)`
- Fixed image height and width
- RGB images normalized to the range `0–1`
- A single binary output score per image
- The positive class represents `PNEUMONIA`

Preprocessing and class interpretation must match the model's training configuration. These assumptions are not yet validated automatically.

## Running the Application

From the project directory, run:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Select an image and click **Classify X-ray**. The page displays the classification label and model score.

The application currently runs with Flask's development server and debug mode enabled. A production WSGI server is planned before deployment.

## Request Flow

```text
Browser uploads an image
          |
          v
app.py receives POST /predict
          |
          v
preprocessing.py prepares the image
          |
          v
inference.py runs the model and selects a label
          |
          v
app.py renders index.html with the result
```

### Image preprocessing

Uploaded images are:

1. Decoded from the uploaded bytes
2. Converted to RGB
3. Resized to the model's expected width and height
4. Converted to a NumPy array with dtype `float32`
5. Given a batch dimension
6. Normalized by dividing pixel values by `255`

The resulting array has shape:

```text
(1, height, width, 3)
```

For a model expecting 256 × 256 RGB images, this becomes:

```text
(1, 256, 256, 3)
```

### Prediction

`PredictionService` loads the model during application startup and reuses it for requests within that process.

The application interprets the first output value using this threshold:

| Score | Classification |
| --- | --- |
| `< 0.5` | NORMAL |
| `>= 0.5` | PNEUMONIA |

The score displayed on the HTML page is rounded to four decimal places; the API returns the score without this rounding. The score is a model output, not a clinically validated probability or confidence percentage.

## Testing

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run automated tests:

```bash
python -m pytest -q
```

Tests use generated images and a fake predictor. They do not require the
trained model and do not evaluate model accuracy.

## Upload Limits

Only decoded PNG and JPEG images are accepted.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MAX_CONTENT_LENGTH` | `10485760` bytes | Maximum total request size |
| `MAX_IMAGE_PIXELS` | `16000000` pixels | Maximum source image pixel count |

Both settings can be overridden through environment variables.

Invalid images return HTTP 400. Requests exceeding the size limit
return HTTP 413.

## Prediction API

### POST `/api/v1/predict`

Submit an image using multipart form data with a field named `file`.

```bash
curl -F "file=@/path/to/xray.png" \
  http://127.0.0.1:5000/api/v1/predict
```

On PowerShell, use `curl.exe`.

Example successful response:

```json
{
  "label": "NORMAL",
  "score": 0.123456,
  "model_version": "cnn_best_100"
}
```

`MODEL_VERSION` is a configurable model identifier. It defaults to
`cnn_best_100` and should be updated when using a different model.

| Status | Meaning |
| --- | --- |
| 200 | Prediction completed |
| 400 | Missing or invalid image |
| 413 | Request exceeds the upload limit |

Validation errors return:

```json
{
  "error": {
    "code": "invalid_image",
    "message": "Please select an image file."
  }
}
```

Oversized requests use the error code `upload_too_large`.

The API uses the same preprocessing and predictor as the HTML upload page.

## Current Limitations

- Image validation checks format, readability, and size; it does not verify that an uploaded image is a chest X-ray.
- Unexpected model inference failures do not yet have dedicated error handling.

## Development Roadmap

- [x] Flask web application
- [x] TensorFlow/Keras model inference
- [x] Browser-based image upload
- [x] Styled results and missing-upload errors
- [x] Extract image preprocessing into a Python module
- [x] Extract model inference into a prediction service
- [x] Configure model location through `MODEL_PATH`
- [x] Introduce a Flask application factory
- [x] Add image validation and upload limits
- [x] Add automated tests with pytest
- [x] Add a JSON prediction API
- [ ] Add health checks and request logging
- [ ] Automate tests and linting with GitHub Actions
- [ ] Containerize the application with Docker
- [ ] Configure a production WSGI server
- [ ] Establish versioned model delivery for deployment
- [ ] Deploy to local Kubernetes using kind
- [ ] Add a Python deployment verification CLI
- [ ] Document and rehearse updates, recovery, and rollback

## Disclaimer

This project is for educational and demonstration purposes only. It is not a clinically validated medical diagnostic system and should not be used to make medical decisions.

Chest X-ray interpretation and diagnosis should be performed by qualified healthcare professionals.

