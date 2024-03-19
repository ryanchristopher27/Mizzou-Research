FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

WORKDIR /data

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# COPY practice_notebook.ipynb practice_notebook.ipynb
COPY ucmerced_classification.py ucmerced_classification.py

COPY Images Images

# COPY . .

# COPY cifar10_classification.py cifar10_classification.py

# RUN jupyter nbconvert --to python practice_notebook.ipynb
