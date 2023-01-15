from flask import Flask, request, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
import marshal
from pathlib import Path
import numpy as np
from tensorflow import keras
import requests
from PIL import Image
import random
import json
import urllib

def load_model(path):
    file = Path(path)
    if not file.exists():
        raise Exception("File not found")
    with open(path, "rb") as f:
        import types
        serialized = marshal.loads(f.read())
        predict = types.FunctionType(serialized, globals(), "predict")
    return predict


def img_to_array(img):
    byte_arr = BytesIO()
    img.save(byte_arr, format=img.format)
    byte_arr = byte_arr.getvalue()
    return byte_arr


def predict(model, text):
    """Returns a PIL image of the text, can be converted to BytesIO if needed"""
    """Long texts will take a long time to process"""
    img = model(text)
    filename = f"{random.randint(1, 100000000000)}.jpg"
    img.save(f"./imgs/{filename}")
    return {"data": [{"url": f"{request.url.replace('/main', '/imgs')}/{filename}"}]}


app = Flask(__name__)


@app.route('/main', methods=['POST'])
def main_screen():
    print(json.loads(request.get_data()))
    if request.method == 'POST':
        text = json.loads(request.get_data())["text"]
        return predict(load_model("./serialized_bin"), text)


@app.route('/imgs/<filename>', methods=['GET'])
def get_img(filename):
    return send_from_directory("imgs", filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=2500)
