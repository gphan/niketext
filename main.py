#!/usr/bin/env python3

from flask import Flask
from flask import render_template, request
from textgenrnn import textgenrnn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    temperature = request.args.get('temperature', default=0.5, type=float)
    samples = request.args.get('samples', default=2, type=int)
    textgen = textgenrnn("niketext_weights.hdf5")
    text = textgen.generate(n=samples, temperature=temperature, return_as_list=True)
    return "\n".join(text)

if __name__ == '__main__':
    app.run()