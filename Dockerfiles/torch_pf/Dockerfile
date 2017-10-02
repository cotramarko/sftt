# Base image
FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

RUN pip3 install matplotlib
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip3 install torchvision
RUN apt-get update
RUN apt-get install -y python3-tk