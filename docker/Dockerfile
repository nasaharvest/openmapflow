# syntax = docker/dockerfile:experimental
FROM pytorch/torchserve:0.5.3-cpu
USER root
RUN pip3 install --upgrade pip
RUN pip install openmapflow[data]==0.1.6rc3
COPY torchserve_handler.py /home/model-server/handler.py
ADD torchserve_start.sh /usr/local/bin/start.sh
RUN chmod 777 /usr/local/bin/start.sh
