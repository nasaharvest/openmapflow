# syntax = docker/dockerfile:experimental
FROM pytorch/torchserve:0.6.0-cpu
USER root
RUN pip3 install --upgrade pip
ARG VERSION
RUN pip install openmapflow[data]==${VERSION}
RUN cp $(openmapflow dir)/torchserve_handler.py /home/model-server/handler.py
RUN cp $(openmapflow dir)/scripts/torchserve_start.sh /usr/local/bin/start.sh
RUN chmod 777 /usr/local/bin/start.sh

# Ensures that everytime models.dvc is updated
# This following docker steps are rerun
ARG MODELS_DIR
COPY $MODELS_DIR.dvc /home/model-server
COPY $MODELS_DIR/*.pt /home/model-server/

WORKDIR /home/model-server

ARG MODELS
RUN for m in $MODELS; \
    do torch-model-archiver \
    --model-name $m \
    --version 1.0 \
    --serialized-file $m.pt \
    --handler handler.py \
    --export-path=model-store; \
    done

ARG DEST_BUCKET
ENV DEST_BUCKET ${DEST_BUCKET}
ENV MODELS ${MODELS}
CMD ["/usr/local/bin/start.sh", "\"${MODELS}\""]


