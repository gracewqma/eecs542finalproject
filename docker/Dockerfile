FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV TZ=US
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


# Update pip
RUN python -m pip install --upgrade setuptools==59.5.0 pip

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install -r /workspace/requirements.txt