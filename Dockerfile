# first stage
FROM python:3.8 AS builder
COPY requirements.txt .

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y \
    git \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-pip \
    gcc \
    python3-tk \
    ffmpeg \
    libopenblas-dev \
    liblapack-dev

# Install dlib
RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . && \
    cd .. && \
    python3 setup.py install

# Install Face Recognition and OpenCV
RUN pip3 install face_recognition opencv-python
RUN pip install --user -r requirements.txt


FROM python:3.8-slim
WORKDIR /code

COPY --from=builder /root/.local /root/.local
# COPY ./FaceRecognition .

ENV PATH=/root/.local:$PATH

CMD ["python", "-u", "./main.py", "./data_encidings.pickle"]