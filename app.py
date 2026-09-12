from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image

app = Flask(__name__)

# Load model once when the application starts
model = load_model("model/cnn_best_100.h5")

# Get model input dimensions dynamically from model
input_height = model.input_shape[1]
input_width  = model.input_shape[2]

print(f"Model input size: {input_width} x {input_height}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please select an image file.")

    file = request.files["file"]

    # Prepare the uploaded image for the model
    img_array = preprocess_image(
        file.read(),
        width=input_width,
        height=input_height,
    )

    # Predict
    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])

    # Classify
    label = "PNEUMONIA" if score >= 0.5 else "NORMAL"

    return render_template("index.html", label=label, score=round(score, 4))


if __name__ == "__main__":
    app.run(debug=True)
