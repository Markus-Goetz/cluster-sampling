FROM tensorflow/tensorflow:1.8.0-gpu-py3
MAINTAINER Markus Goetz "markus.v.goetz@gmail.com"

RUN pip install --no-cache-dir keras
RUN rm -rf /root/.cache/pip/*
RUN rm -rf /tmp/*

WORKDIR /home
