import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

def preprocess(img):
    print("preprocess: ", img.shape, img.dtype)

    img = img / 255.

    img = cv2.resize(img, (256, 256))

    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    print(y0, x0)
    img = img[y0 : y0+224, x0 : x0+224, :]

    img = (img - [0.485, 0.456, 0.406]) 
    img = img / [0.229, 0.224, 0.225]
    
    img = np.transpose(img, axes=[2, 0, 1])    
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return img

def postprocess(preds):
    print("preds: ", preds.shape, preds.dtype)
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    print("a[0]: ", a[0])
    print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))
    return labels[a[0]]