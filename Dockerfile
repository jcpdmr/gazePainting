# Not working

FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04

ENV TZ=Europe/Rome

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y

RUN apt update -y

RUN apt-get install -y python3 python3-pip sudo

RUN sudo apt-get install git-all -y

RUN sudo apt install -y nvidia-cuda-toolkit

RUN sudo mkdir home/gazePainting

COPY . home/gazePainting

WORKDIR home/gazePainting

RUN apt install python3.8-venv

RUN python3 -m venv venv

RUN pip install -r other_files/requirements.txt

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install git+https://github.com/elliottzheng/face-detection.git@master



