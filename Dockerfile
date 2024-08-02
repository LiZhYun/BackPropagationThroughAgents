# ARG DOCKER_BASE
# FROM $DOCKER_BASE

# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
#   libgl1-mesa-dev libsdl2-dev \
#   libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
#   libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
#   libboost-python-dev wget libosmesa6-dev libgl1-mesa-glx libglfw3

# # RUN mkdir -p ~/miniconda3
# # RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# # RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# # RUN rm -rf ~/miniconda3/miniconda.sh

# # RUN ~/miniconda3/bin/conda init bash
# # RUN ~/miniconda3/bin/conda init zsh
# RUN echo "conda activate base" > ~/.bashrc
# # RUN pip install --upgrade pip

# RUN conda install wheel==0.38.4
# RUN conda install setuptools==65.5.0
# RUN conda install psutil
# RUN conda install -y anaconda::py-boost
# # RUN conda install -c conda-forge glew
# # RUN conda install -y -c conda-forge libstdcxx-ng
# # RUN conda install -c conda-forge mesalib
# # RUN conda install -c menpo glfw3

# COPY . /gfootball
# RUN cd /gfootball && pip install .
# WORKDIR '/gfootball'

ARG DOCKER_BASE
FROM $DOCKER_BASE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  python3-pip 

RUN ln -snf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/bin/../lib/libstdc++.so.6

RUN python3 -m pip install --upgrade pip 
RUN python3 -m pip install psutil
RUN python3 -m pip install setuptools==66
RUN python3 -m pip install wheel==0.38.4
RUN python3 -m pip install importlib-metadata==4.13.0

# RUN echo "export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH" > ~/.bashrc

COPY . /gfootball
RUN cd /gfootball && python3 -m pip install .
WORKDIR '/gfootball'