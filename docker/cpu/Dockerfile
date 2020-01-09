# base nvidia cuda image
FROM continuumio/miniconda3

# install pytorch
RUN conda install pytorch cpuonly -c pytorch

# install python libs
RUN conda install \
    jupyter \
    numpy \
    matplotlib 

RUN pip --no-cache-dir install \
    --upgrade pip \
    fastprogress \
    jupyter_contrib_nbextensions

# install some jupyter notebook plugins
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable collapsible_headings/main
RUN jupyter nbextension enable scroll_down/main
RUN jupyter nbextension enable execute_time/ExecuteTime
RUN jupyter nbextension enable livemdpreview/livemdpreview
RUN jupyter nbextension enable spellchecker/main

# set workspace as default working directory
WORKDIR /workspace
RUN chmod -R a+w .