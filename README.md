# Pediatric Pneumonia Detector

A Python and Flask web application that uses a TensorFlow/Keras convolutional neural network (CNN) to classify pediatric chest X-ray images as **NORMAL** or **PNEUMONIA**.

This project combines machine learning inference with a Python web application and is intended as an educational and portfolio project.

## Features

* Upload chest X-ray images through a web interface
* TensorFlow/Keras CNN inference
* Automatic image preprocessing
* Dynamic image resizing based on the trained model input dimensions
* Pixel normalization using `1/255`
* Binary classification using a 0.5 threshold
* Flask-based web application

## Technology Stack

* Python 3.12
* Flask
* TensorFlow / Keras
* NumPy
* Pillow
* HTML
* Git

## Project Structure

```text
pediatric-pneumonia-detector/
├── app.py
├── model/
│   └── <trained-model>.h5
├── templates/
│   └── index.html
├── requirements.txt
├── README.md
└── .gitignore
```

> The trained model is kept private and is excluded from the Git repository using `.gitignore`.

## Installation

Clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/pediatric-pneumonia-detector.git
cd pediatric-pneumonia-detector
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model

The trained TensorFlow/Keras model is **not included in this repository**.

The application expects a trained model to be available locally in the:

```text
model/
```

directory.

The model is kept private and must be available locally for the application to perform predictions.

## Running the Application

Start the Flask application:

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

Open the address in a web browser and upload a supported chest X-ray image.

## Model Input

The trained CNN expects an input shape of:

```text
256 × 256 × 3
```

Uploaded images are:

1. Converted to RGB
2. Resized to the model's expected dimensions
3. Converted to a NumPy array
4. Normalized by dividing pixel values by `255`
5. Passed to the trained CNN for inference

## Prediction

The model produces a score between `0` and `1`.

|    Score | Classification |
| -------: | -------------- |
|  `< 0.5` | NORMAL         |
| `>= 0.5` | PNEUMONIA      |

Example:

```text
0.0 ───────────── 0.5 ───────────── 1.0
       NORMAL             PNEUMONIA
```

## Development Roadmap

* [x] Flask web application
* [x] TensorFlow/Keras model inference
* [x] Image preprocessing
* [x] Browser-based image upload
* [x] Model input validation
* [ ] Automated tests with pytest
* [ ] Docker containerization
* [ ] Docker Compose configuration
* [ ] GitHub Actions CI
* [ ] Application health check
* [ ] Production WSGI server
* [ ] Improved error handling

## Disclaimer

This project is for educational and demonstration purposes only. It is not a clinically validated medical diagnostic system and should not be used to make medical decisions.

Chest X-ray interpretation and diagnosis should be performed by qualified healthcare professionals.

