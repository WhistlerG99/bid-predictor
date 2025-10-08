# Dockerfile
FROM python:3.11-slim

# non-root for safety
RUN useradd -m -s /bin/bash trainer
WORKDIR /opt/program

# system deps: compilers + libs for matplotlib headless and scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# SageMaker training toolkit (lets the container run your entry_point & channels)
RUN pip install --no-cache-dir "sagemaker-training>=4,<5"

# copy just the requirements first for better layer cache
COPY requirements.txt ./requirements.txt
# your requirements do NOT include matplotlib; you import it in train.py
# install your stack + matplotlib (Agg backend works headless)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "matplotlib>=3.9"

# optional: if you ship a package (you do), you can pre-install it
# COPY pyproject.toml ./
# COPY bid_predictor ./bid_predictor
# RUN pip install --no-cache-dir .

# default program if no entry_point is given; SDK can override this
ENV SAGEMAKER_PROGRAM=train.py
ENV MPLBACKEND=Agg

USER trainer

# hint to SageMaker that this is a pure training image
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=false

# Switch to root to prepare SageMaker directories
USER root
# Create the directories SageMaker expects and hand them to 'trainer'
RUN mkdir -p /opt/ml/code /opt/ml/input /opt/ml/model /opt/ml/output \
 && chown -R trainer:trainer /opt/ml /opt/program

# (optional, extra safety) make group-writable too
# RUN chgrp -R 0 /opt/ml /opt/program && chmod -R g=u /opt/ml /opt/program

# Back to non-root
USER trainer