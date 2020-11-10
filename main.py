#!/usr/bin/env python3

from flask import Flask
from flask import render_template
from textgenrnn import textgenrnn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    textgen = textgenrnn("niketext_weights.hdf5")
    text = textgen.generate(n=10, return_as_list=True)
    return "\n".join(text)

if __name__ == '__main__':
    app.run(threaded=False)