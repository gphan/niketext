# Nike Text Generator
Generates Nike marketing text from a trained textgenrnn model.

# Running Locally

## Prerequisites

- Python 3.8

## Setup
Either setup a venv first or install fresh:

```
$ pip3 install -r requirements.txt
```

## Running

```
$ python3 main.py
```

## Testing

Go to http://localhost:5000 in your browser or use the API below:

```
$ curl localhost:5000/generate?samples=1&prefix=nike
```
