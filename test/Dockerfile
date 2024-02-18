FROM nvcr.io/nvidia/tensorrt:23.08-py3
WORKDIR /traffic_light

RUN DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install python3-dev python3-numpy -y
# RUN apt-get install libopencv-dev -y

RUN pip3 install fastapi
RUN pip3 install uvicorn==0.16.0
RUN pip3 install filterpy
RUN pip3 install requests
RUN pip3 install PyYAML
RUN pip3 install pydantic==1.10.2
RUN pip3 install numpy==1.22.4

RUN pip3 install lap
RUN pip3 install motmetrics scikit-image Pillow
RUN pip3 install Cython
RUN pip3 install cython_bbox
RUN pip3 install shapely
RUN pip3 install onnx==1.12.0
RUN pip3 install python-socketio
RUN pip3 install pycuda

RUN pip3 install opencv-python==4.7.0.72
RUN pip3 install requests
RUN pip3 install psutil
RUN pip3 install shapely
RUN pip3 install python-multipart
RUN pip3 install faiss-gpu