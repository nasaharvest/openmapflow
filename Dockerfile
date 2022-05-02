# syntax = docker/dockerfile:experimental
FROM pytorch/torchserve:0.5.3-cpu as base

USER root

FROM base as reqs
RUN pip3 install --upgrade pip
RUN pip install cropharvest==0.3.0 google-cloud-storage netCDF4 pandas rasterio xarray

FROM reqs as build-torchserve
COPY openmapflow/torchserve_handler.py /home/model-server/handler.py

ADD openmapflow/torchserve_start.sh /usr/local/bin/start.sh
RUN chmod 777 /usr/local/bin/start.sh


