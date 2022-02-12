FROM nvidia/cuda:11.3.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
    sudo \
    git \
    vim

RUN apt install -y python3.8 python3.8-distutils curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py

COPY pytorch.txt pytorch.txt

RUN pip install -r pytorch.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY requirements.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid ${GROUP_ID} ubuntu
RUN useradd -rm \
            -d /home/ubuntu \
            -s /bin/bash \
            -g ubuntu \ 
            -G sudo \
            -u ${USER_ID} \
            -p "$(openssl passwd -1 ubuntu)" \
            ubuntu

USER ubuntu
RUN echo "alias python=python3.8" > /home/ubuntu/.bashrc