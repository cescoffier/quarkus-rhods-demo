import random
import numpy as np
import cv2
import argparse
import grpc
import grpc_predict_v2_pb2_grpc
import grpc_predict_v2_pb2
import time
from model import preprocess, postprocess

class client:
    def __init__(self, grpc_host, grpc_port, model_name):
        self.host = grpc_host
        self.port = grpc_port
        self.model_name = model_name
        # options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(self.channel)
        print("gRPC client instantiated: ", self.host, self.port, self.model_name)

    def __call__(self, img):
        """
        Makes a prediction on a given image by calling an inference endpoint served by ModelMesh.
        """
        data = preprocess(img)
        
        # request content building
        inputs = []
        inputs.append(grpc_predict_v2_pb2.ModelInferRequest().InferInputTensor())
        inputs[0].name = "data"
        inputs[0].datatype = "FP32"
        inputs[0].shape.extend([1, 3, 224, 224])
        arr = data.flatten()
        inputs[0].contents.fp32_contents.extend(arr)

        # request building
        request = grpc_predict_v2_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.inputs.extend(inputs)

        # Call the gRPC server and get the response
        try:
            response = self.stub.ModelInfer(request)
        except grpc.RpcError as e:
            raise Exception(f"Failed to call gRPC server: {e.details()}")
        
        # unserialize response content
        result_arr = np.frombuffer(response.raw_output_contents[0], dtype=np.float32)
        for byte in response.raw_output_contents[0][:10]:
            print(f'{byte:02X}', end=' ')   

        out = postprocess(result_arr)
        
        return out
    
    def invoke(self, img):
        inputs = []
        inputs.append(grpc_predict_v2_pb2.ModelInferRequest().InferInputTensor())
        inputs[0].name = "data"
        inputs[0].datatype = "FP32"
        inputs[0].shape.extend([1, 3, 224, 224])
        arr = img.flatten()
        inputs[0].contents.fp32_contents.extend(arr)

        # request building
        request = grpc_predict_v2_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.inputs.extend(inputs)

        # Call the gRPC server and get the response
        try:
            response = self.stub.ModelInfer(request)
        except grpc.RpcError as e:
            raise Exception(f"Failed to call gRPC server: {e.details()}")
        # unserialize response content
        
        test = np.frombuffer(response.raw_output_contents[0], dtype=np.float32);
        print(test.shape, test.dtype, test.strides)
        # print(test)
        
        print(type(response.raw_output_contents[0]))
        for i in range(10):
            print(response.raw_output_contents[0][i])
        
        return response.raw_output_contents[0]
        # return np.frombuffer(response.raw_output_contents[0], dtype=np.float32)

    