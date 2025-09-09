from flask import Flask, request
from model import generate
import pickle

generate()

ai = pickle.load('ai.pkl', 'rb')

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Server Runnionggggggggggggggggggg"

@app.route('/predict')
def predict():
    ir = request.args.get('ir')
    ir = int(ir)
    data = [[ir]]
    result = ai.predict(data)[0]
    return result


if (__name__) == "__main__":
    app.run(host = '0.0.0.0',port=3000)