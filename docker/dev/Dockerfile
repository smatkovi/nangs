# base nvidia cuda image
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

# install system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# install miniconda3
ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# install pytorch
RUN conda install pytorch cudatoolkit=10.0 -c pytorch

# install python libs
RUN conda install \
    jupyter \
    numpy \
    matplotlib 

RUN pip --no-cache-dir install \
    --upgrade pip \
    fastprogress \
    nbdev \
    jupyter_contrib_nbextensions

# install some jupyter notebook plugins
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable collapsible_headings/main
RUN jupyter nbextension enable scroll_down/main
RUN jupyter nbextension enable execute_time/ExecuteTime
RUN jupyter nbextension enable livemdpreview/livemdpreview
RUN jupyter nbextension enable spellchecker/main

# install apex for mixed precision trainig
#RUN git clone https://github.com/NVIDIA/apex.git
#WORKDIR /apex
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# set workspace as default working directory
WORKDIR /workspace
RUN chmod -R a+w .