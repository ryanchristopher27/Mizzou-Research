FROM gitlab-registry.nrp-nautilus.io/jhurt/mmdet-v3/mmdet-base:v3.1

WORKDIR /rchristopher/data

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

# COPY src/data src/data
# COPY src/mmpretrain src/mmpretrain

# COPY src/results src/results

# COPY src/mmpretrain_configs src/mmpretrain_configs

# COPY src/code src/code