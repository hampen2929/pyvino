# base
FROM ubuntu:16.04

# openvino
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/15944/l_openvino_toolkit_p_2019.3.334_online.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    git \
    zip \
    unzip \
    curl \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh

# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"

RUN echo "source $INSTALL_DIR/bin/setupvars.sh" >> /root/.bashrc

SHELL ["/bin/bash", "-c"]

# intel python
#Set Variables
ARG TEMP_PATH=/tmp/miniconda
ARG MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
ARG INTEL_PYTHON=intelpython3_core=2019.4
 
#Install commands
RUN apt-get update && \
    apt-get install -y bzip2 ca-certificates
 
#Install miniconda3
RUN mkdir -p ${TEMP_PATH} && cd ${TEMP_PATH} && \
    wget -nv  ${MINICONDA_URL} -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda

RUN rm -rf ${TEMP_PATH}
ENV PATH /opt/conda/bin:$PATH
 
# Install Intel Python 3 Full Package
ENV ACCEPT_INTEL_PYTHON_EULA=yes
RUN conda create -n idp python==3.6.8 -y
RUN source activate idp
RUN conda config --add channels intel \
    && conda install -y -q ${INTEL_PYTHON} python=3.6.8 \
    && conda clean --all \
    && apt-get update -qqq \
    && apt-get install -y -q g++ \
    && apt-get autoremove
RUN conda install numpy -c intel --no-update-deps
 
CMD ["/bin/bash"]
