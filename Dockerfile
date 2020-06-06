FROM ubuntu:18.04

# init
USER root
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget cpio sudo git zip unzip curl xterm vim bzip2 ca-certificates lsb-release && \
    rm -rf /var/lib/apt/lists/*

# ubuntu
WORKDIR /workspace/

RUN useradd -m -s /bin/bash ubuntu
RUN gpasswd -a ubuntu sudo
RUN gpasswd -a ubuntu video

USER ubuntu 
WORKDIR /workspace/

################# dev #################

USER root

# intel python
#Set Variables
ARG TEMP_PATH=/tmp/miniconda
ARG MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
ARG INTEL_PYTHON=intelpython3_core=2019.4
  
#Install miniconda3
RUN mkdir -p ${TEMP_PATH} && cd ${TEMP_PATH} && \
    wget -nv  ${MINICONDA_URL} -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda

RUN rm -rf ${TEMP_PATH}
ENV PATH /opt/conda/bin:$PATH
 
# Install Intel Python 3 core Package
ENV ACCEPT_INTEL_PYTHON_EULA=yes
RUN conda create -n idp python==3.6.8 -y
RUN conda config --add channels intel \
    && conda install -y -q ${INTEL_PYTHON} python=3.6.8 \
    && conda clean --all \
    && apt-get update -qqq \
    && apt-get install -y -q g++ \
    && apt-get autoremove

SHELL ["/bin/bash", "-c"]

# install packages
RUN conda install numpy -c intel --no-update-deps
RUN pip install jupyter 
RUN mkdir /home/ubuntu/.jupyter/
RUN touch /home/ubuntu/.jupyter/jupyter_notebook_config.py

RUN echo $'c.NotebookApp.ip = "*" \n\
c.NotebookApp.notebook_dir = "/workspace/" \n\
c.NotebookApp.port = 8888 \n\
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager" \n\
c.ContentsManager.default_jupytext_formats = "ipynb,py" \n'\
>> /home/ubuntu/.jupyter/jupyter_notebook_config.py

RUN sudo chown ubuntu:ubuntu /home/ubuntu/.jupyter -R

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
    && sudo mv cmake-*-Linux-x86_64 /opt \
    && sudo ln -s /opt/cmake-3.16.2-Linux-x86_64/bin/* /usr/bin

# openvino
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16670/l_openvino_toolkit_p_2020.3.194_online.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

RUN mkdir -p $TEMP_DIR && \
    cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    ls && \
    cd l_openvino_toolkit_p_2020.3.194_online && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

RUN /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh
RUN source /opt/intel/openvino/bin/setupvars.sh
RUN echo $'source /opt/intel/openvino/bin/setupvars.sh' >> ~/.bashrc

# USER ubuntu
CMD ["/bin/bash"]
RUN chown ubuntu:ubuntu /workspace/ -R
RUN chown ubuntu:ubuntu /opt/conda/lib/python3.6/site-packages/ -R
RUN chown ubuntu:ubuntu /opt/conda/bin/ -R
RUN echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN visudo --c

USER ubuntu 
WORKDIR /workspace
