# base nvidia cuda image
FROM continuumio/miniconda3

# install pytorch
RUN conda install pytorch cpuonly -c pytorch

# install python libs
RUN conda install \
    numpy \
    matplotlib 

RUN pip --no-cache-dir install \
    --upgrade pip \
    nangs \
    fastprogress 

# set workspace as default working directory
WORKDIR /workspace
RUN chmod -R a+w .