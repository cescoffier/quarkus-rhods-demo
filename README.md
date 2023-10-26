# Demo of Red Hat OpenShift Data Science 


This demo invokes a model served used Red Hat OpenShift Data Science Model Serving.
The model used in this demo is `resnet50`, and use ONNX model format.

## Prerequisites

First, you need an instance of Openshift serving your model. 
Then, the 8033 port of that instance should be port-forwarded to localhost (to simplify the test).

## Run the demo

```
mvn quarkus:dev
```

Then, send your picture to the endpoint:

```
POST http://localhost:8080/hello
Content-Type: multipart/form-data; boundary=WebAppBoundary

--WebAppBoundary
Content-Disposition: form-data; name="image"; filename="image.jpg"

< ./cat-rabbit-dog.jpeg
--WebAppBoundary--
```

It returns the 3 top-objects.
