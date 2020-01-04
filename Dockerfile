# base
FROM ubuntu:18.04

# ubuntu setting
# RUN useradd -m -s /bin/bash ubuntu

# RUN echo 'ubuntu:ubuntu' |chpasswd
# RUN gpasswd -a ubuntu sudo
# USER ubuntu
WORKDIR /home/ubuntu/

USER root
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
    xterm \
    vim \
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

RUN echo "alias openvino='source /opt/intel/openvino/bin/setupvars.sh'" >> ~/.bashrc

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
 
# Install Intel Python 3 core Package
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
# env
RUN echo source activate idp  >> ~/.bashrc

# jupyter notebook
RUN pip install jupyter
RUN jupyter notebook --generate-config
RUN ipython kernel install --user --name=idp --display-name=idp

# CMAKE
RUN apt-get update
RUN sudo apt remove cmake -y
ARG DOWNLOAD_LINK=https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-Linux-x86_64.sh
ARG TEMP_DIR=/home/ubuntu/cmake_installer/cmake-3.16.2-Linux-x86_64/

RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR \
    && wget $DOWNLOAD_LINK \
    && chmod +x cmake-*-Linux-x86_64.sh \
    && sudo bash cmake-*-Linux-x86_64.sh --skip-license \
    && cd .. \
    && ls \
    && sudo mv cmake-*-Linux-x86_64 /opt \
    && sudo ln -s /opt/cmake-3.16.2-Linux-x86_64/bin/* /usr/bin

# export PATH=TEMP_DIR/bin:$PATH

# env path
RUN echo LC_ALL=C.UTF-8  >> ~/.bashrc
RUN echo LANG=C.UTF-8  >> ~/.bashrc

# pyvino
ENV PYTHONPATH /home/ubuntu/src_dir/pyvino/pyvino/model/human_pose_estimation/human_3d_pose_estimator/pose_extractor/build/:$PYTHONPATH
RUN echo openvino >> /root/.bashrc

# USER ubuntu
CMD ["/bin/bash"]
