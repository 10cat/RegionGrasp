FROM nvidia/cuda:11.4.0-base-ubuntu18.04
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update
RUN apt install -y wget

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh && \
    /bin/bash ./Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Anaconda3-2022.10-Linux-x86_64.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN /opt/conda/bin/pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html


RUN /opt/conda/bin/pip install git+https://github.com/YiLin32Wang/MANO.git

RUN /opt/conda/bin/pip install git+https://github.com/facebookresearch/pytorch3d.git

RUN /opt/conda/bin/pip install git+https://github.com/otaheri/chamfer_distance

RUN /opt/conda/bin/pip install matplotlib

RUN /opt/conda/bin/pip install wandb

RUN /opt/conda/bin/pip install omegaconf

RUN /opt/conda/bin/pip install timm

RUN /opt/conda/bin/pip install scikit-learn

RUN /opt/conda/bin/pip install pybullet

RUN /opt/conda/bin/pip install imageio

RUN /opt/conda/bin/pip install numpy==1.23.4







