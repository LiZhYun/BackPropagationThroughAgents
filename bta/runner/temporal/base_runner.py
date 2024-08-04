# ARG DOCKER_BASE
# FROM $DOCKER_BASE

# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
#   libgl1-mesa-dev libsdl2-dev \
#   libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
#   libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
#   python3-pip

# RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install psutil wandb igraph tensorboardX imageio icecream easydict scipy python-dateutil
# RUN python3 -m pip install setuptools==66
# RUN python3 -m pip install wheel==0.38.4
# RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# COPY . /gfootball
# RUN cd /gfootball && python3 -m pip install .
# WORKDIR '/gfootball'

ARG DOCKER_BASE
FROM $DOCKER_BASE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libosmesa6-dev libgl1-mesa-glx libglfw3 \
  python3-pip 

# RUN ln -snf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/bin/../lib/libstdc++.so.6
# RUN ln -snf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/python3.10/site-packages/torch/lib/../../../../libstdc++.so.6

RUN python3 -m pip install --upgrade pip 
RUN python3 -m pip install psutil wandb igraph tensorboardX imageio icecream easydict scipy python-dateutil
RUN python3 -m pip install setuptools==66
RUN python3 -m pip install wheel==0.38.4
RUN python3 -m pip install git+https://github.com/oxwhirl/smacv2.git
# RUN python3 -m pip install importlib-metadata==4.13.0
# RUN conda install -y anaconda::py-boost
# RUN conda install -c conda-forge libstdcxx-ng

# RUN echo "export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH" > ~/.bashrc

COPY . /gfootball
RUN cd /gfootball && python3 -m pip install .
WORKDIR '/gfootball'
