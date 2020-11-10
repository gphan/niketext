#!/usr/bin/env python3

import os
from flask import Flask
from flask import render_template, request, send_from_directory
from textgenrnn import textgenrnn

app = Flask(__name__)
textgen = textgenrnn(weights_path="niketext_weights.hdf5",
                     vocab_path="niketext_vocab.json",
                     config_path="niketext_config.json")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['GET'])
def generate():
    temperature = request.args.get('temperature', default=0.5, type=float)
    samples = request.args.get('samples', default=2, type=int)
    prefix = request.args.get('prefix', default=None, type=str)
    text = textgen.generate(n=samples, temperature=temperature, prefix=prefix, return_as_list=True)
    return "\n".join(text)


if __name__ == '__main__':
    app.run()
