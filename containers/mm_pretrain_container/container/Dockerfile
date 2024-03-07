FROM gitlab-registry.nrp-nautilus.io/jhurt/mmdet-v3/mmdet-base:v3.1

WORKDIR /data

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .
