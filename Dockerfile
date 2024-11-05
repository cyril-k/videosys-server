# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Uninstall apex first
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# RUN pip install git+https://github.com/NUS-HPC-AI-Lab/VideoSys
RUN git clone https://github.com/NUS-HPC-AI-Lab/VideoSys /VideoSys \
    && cd /VideoSys \
    && pip install -e .
RUN pip install fastapi["standard"]==0.115.4 \
    prometheus-client==0.21.0 \
    boto3==1.35.50

COPY ./app /VideoSys/app
ENV PYTHONPATH "${PYTHONPATH}:/VideoSys/app" 
# COPY ./config /config
RUN mkdir /results

WORKDIR /VideoSys/app

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]