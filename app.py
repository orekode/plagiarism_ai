from flask import Flask, request, jsonify
from model import Plagiarism
import math

app = Flask(__name__)

detector = Plagiarism()

@app.route('/train', methods=['POST'])
def train():
    text = request.get_json()['text']
    detector.train(text)
    return jsonify({'message': 'Model trained successfully'})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text = request.get_json()['text']
    prediction = detector.calculate_similarity(text)
    return jsonify({'prediction': math.floor(prediction)})


if __name__ == '__main__':
    app.run(debug=True)