from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model.pkl")

# Store detection history
history = []


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    message = None

    if request.method == "POST":
        message = request.form["message"]

        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]
        confidence = round(max(probabilities) * 100, 2)

        if prediction == "spam":
            result = "SPAM MESSAGE"
        else:
            result = "NOT SPAM"

        history.insert(0, {
            "message": message,
            "result": result,
            "confidence": confidence
        })

        # Keep only latest 10 detections
        if len(history) > 10:
            history.pop()

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message=message
    )


@app.route("/history")
def show_history():
    return render_template("history.html", history=history)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)