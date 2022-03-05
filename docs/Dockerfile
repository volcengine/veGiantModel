FROM nvcr.io/nvidia/pytorch:21.05-py3 

RUN pip3 install boto3 regex tensorboardX==1.8 wheel pybind11 ninja psutil pyprof
RUN apt-get -yq autoremove --purge ibverbs-providers
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends --allow-downgrades \
     libibverbs-dev=28.0-1ubuntu1 libibverbs1=28.0-1ubuntu1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends --allow-downgrades \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        openssh-client \
        openssh-server \
        ibverbs-providers \
        libibverbs-dev=28.0-1ubuntu1 \
        librdmacm-dev \
        vim \
        iputils-ping \
        llvm-10-dev \
        iproute2 \
        unzip

RUN ln -s /usr/bin/aclocal-1.16 /usr/local/bin/aclocal-1.14
RUN ln -s /usr/bin/automake /usr/local/bin/automake-1.14

ENV LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
ENV BYTEPS_WITH_UCX 0

#install byteps from package stored in tos at volcengine
# RUN pip3 install https://giant-model-package.tos-cn-beijing.volces.com/byteps-0.7.2-cp38-cp38-linux_x86_64.whl

#install byteps from source
RUN git clone --recursive -b bccl-github https://github.com/bytedance/byteps.git && \
    cd byteps && python3 setup.py install

WORKDIR /root