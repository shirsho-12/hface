from flask import Flask, request, send_file
from io import BytesIO
import marshal
from pathlib import Path

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
    return send_file(
        BytesIO(img_to_array(img)),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='test.jpg')


app = Flask(__name__)

@app.route('/main', methods=['POST'])
def main_screen():
    if request.method == 'POST':
        text = request.args.get("text")
        return predict(load_model("./serialized_bin.uu"), text)


if __name__ == '__main__':
    app.run()
