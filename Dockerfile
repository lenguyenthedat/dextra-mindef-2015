FROM ubuntu:12.04

RUN apt-get update && apt-get install -y \
        postgresql \
        git-core \
        build-essential \
        python-matplotlib \
        python2.7 \
        python2.7-dev \
        python-setuptools \
        python-pip \
        python-scipy \
        libatlas-dev \
        libatlas3gf-base

RUN pip install numpy==1.9.2
RUN pip install pandas==0.15.2
RUN pip install scikit-learn==0.16.1

# Install xgboost
RUN cd /usr/local/src && git clone https://github.com/dmlc/xgboost.git  && cd xgboost && sh build.sh && cd python-package && python setup.py install

# Copy source code to container
RUN mkdir -p /root/dextra-mindef-2015
COPY data /root/dextra-mindef-2015/data
COPY classify-xgb-native.py /root/dextra-mindef-2015/
WORKDIR /root/dextra-mindef-2015

# Test run to see if it's working properly
RUN python classify-xgb-native.py
