import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from model import preprocess, postprocess
from remote_inference import client
from flask import Flask, request, jsonify, Response



app = Flask(__name__)


model_path = 'resnet50-v1-12.onnx'
model = onnx.load(model_path)
# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider']).
session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

def get_image(path):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    return img

# For HTTP endpoint
def inference(img):
    ort_inputs = {session.get_inputs()[0].name: img}
    print(session.run(None, ort_inputs))
    preds = session.run(None, ort_inputs)[0]
    print("preds: ", preds.shape, preds.dtype)
    preds2 = np.squeeze(preds)
    a = np.argsort(preds2)[::-1]
    print("a[0]: ", a[0])
    print('class=%s ; probability=%f' %(labels[a[0]],preds2[a[0]]))
    return preds

# Standalone
def predict(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    ort_inputs = {session.get_inputs()[0].name: img}
    print(session.run(None, ort_inputs))
    preds = session.run(None, ort_inputs)[0]
    print("preds: ", preds.shape, preds.dtype)
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    print("a[0]: ", a[0])
    print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))

def grpc(path): 
    img = get_image(path)
    client("localhost", 8033, "test")(img)


@app.route('/inference', methods=['POST'])
def perform_inference():
    try:
        input_data = request.get_json()    

        if not isinstance(input_data, list) or len(input_data) != 3 * 224 * 224:
            return jsonify({"error": "Invalid input data format or size"}), 400

        input_data = np.array(input_data, dtype=np.float32).reshape(1, 3, 224, 224)
        
        result = client("localhost", 8033, "test").invoke(input_data)
        response = Response(result, content_type='application/octet-stream')

        # result = result.flatten().tolist()

        return response, 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)

# Enter path to the inference image below
# img_path = 'kitten.jpg'
# predict(img_path)
grpc('kitten.jpg')
