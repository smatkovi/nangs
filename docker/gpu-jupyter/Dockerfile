# base nvidia cuda image
FROM sensioai/nangs

RUN conda install jupyter

RUN pip --no-cache-dir install \
    --upgrade pip \
    jupyter_contrib_nbextensions

# install some jupyter notebook plugins
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable collapsible_headings/main
RUN jupyter nbextension enable scroll_down/main
RUN jupyter nbextension enable execute_time/ExecuteTime
RUN jupyter nbextension enable livemdpreview/livemdpreview
RUN jupyter nbextension enable spellchecker/main