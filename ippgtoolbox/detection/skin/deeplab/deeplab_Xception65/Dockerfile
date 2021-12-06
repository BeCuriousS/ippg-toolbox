FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update && apt-get install -y \
    # tools
    git \
    vim \
    python-setuptools \
    python-virtualenv \
    graphviz \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \ 
    libxext6 \
    python-pil \
    python-numpy

RUN pip3 install -U pip && pip3 --no-cache-dir install --upgrade \
    pandas \
    h5py \
    pyyaml \
    virtualenv \
    graphviz \
    pydot \
    keras-tqdm \
    opencv-python \
    jupyter \
    matplotlib \
    PrettyTable \
    tmuxp \
    tf-slim==1.0 \
    tqdm \
    imgaug \
    hdf5storage \
    progressbar2
    
    
RUN pip --no-cache-dir install --no-deps \
    keras==2.3.1

ARG USERNAME="any"
ARG USER_UID=1001
ARG USER_GID=1001

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

# COPY dockerexec.sh /
WORKDIR /app/shared/deeplab/
# ENTRYPOINT ["./dockerexec.sh"]