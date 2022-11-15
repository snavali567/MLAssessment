from flask import Flask, request, jsonify, render_template
import model as md

# Create a flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    gb_prediction = md.prediction(json_, "gb_model.pkl")
    return jsonify(gb_prediction)


if __name__ == "__main__":
    print("Running Flask Server: To stop the server use Ctrl + Z")
    app.run(debug=True)

