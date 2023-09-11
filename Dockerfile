FROM ubuntu:20.04

ARG USER=opencda
ARG CARLA_VERSION=0.9.12
ARG ADDITIONAL_MAPS=true
ARG PERCEPTION=true
ARG SUMO=true
ARG OPENCDA_FULL_INSTALL=true

ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV CARLA_VERSION=$CARLA_VERSION
ENV CARLA_HOME=/home/carla
ENV SUMO_HOME=/usr/share/sumo

ENV CLIENT_NUMBER=3
ENV XPOS=81.7194
ENV YPOS=139.51
ENV SCRIPT=opencda_client
ENV PLDM=
ENV ITER=10
ENV MODE=pldm

# Add new user and install prerequisite packages.

WORKDIR /home/OpenCDA

# ---------------------------FOR USING LOCAL VOLUME------------------------------------ 
# Set group ownership of the directory to be used as volume to some GID not used on any actual groups on the host
# -> chown :1024 /data/myvolume
# Change permissions on the directory to give full access to members of the group (read+write+execute)
# -> chmod 775 /data/myvolume
# Ensure all future content in the folder will inherit group ownership
# -> chmod g+s /data/myvolume
# Create a user in the Dockerfile which is member of the group
# -------------------------------------------------------------------------------------
RUN addgroup --gid 1024 mygroup && \
    useradd -m -g 1024 ${USER}


RUN apt-get update && apt-get install -y software-properties-common \
&& apt-get install -y build-essential cmake debhelper git wget curl xdg-user-dirs xserver-xorg libvulkan1 libsdl2-2.0-0 \
libsm6 libgl1-mesa-glx libomp5 pip unzip libjpeg8 libtiff5 software-properties-common nano fontconfig

RUN	add-apt-repository ppa:deadsnakes/ppa && \
	apt-get install -y python3.7 && \
	apt-get install -y ffmpeg libsm6 libxext6 && \
	apt-get install -y python3-pip && \
	apt-get install -y python3.7-distutils && \
	apt-get install -y python3-apt

RUN apt-get install python3-dev

RUN python3.7 -m pip install --upgrade pip && python3.7 -m pip install --upgrade setuptools


COPY requirements.txt .

RUN python3.7 -m pip install -r requirements.txt

RUN python3.7 -m pip install carla==0.9.12

RUN python3.7 -m pip install coloredlogs

RUN python3.7 -m pip install omegaconf

RUN python3.7 -m pip install zmq

RUN python3.7 -m pip install filterpy

RUN python3.7 -m pip install torch torchvision torchaudio yolov5 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# RUN python3.8 -m pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

RUN apt-get install -y libglfw3-dev

RUN python3.7 -m pip install open3d

# Install SUMO.

RUN add-apt-repository ppa:sumo/stable && apt-get update && apt-get install -y sumo sumo-tools sumo-doc \
&& pip install traci

RUN apt-get install -y python-dev 
RUN apt-get install -y libpython3.7-dev libpython3-dev


RUN cd /home/OpenCDA

RUN python3.7 -m pip install python-qpid-proton

# Until I create a repo 
COPY --chown=$USER:$USER . .

# Carla
#EXPOSE 2000/tcp
USER ${USER}

# CMD ["/bin/sh", "-c", "python3.7 opencda.py -t ${SCRIPT} -v 0.9.12 --apply_ml --pmNum ${CLIENT_NUMBER} -x ${XPOS} -y ${YPOS} ${PLDM}"]

CMD ["/bin/sh", "-c", "./run_sim.sh ${ITER} ${SCRIPT} ${MODE}"]

# CMD ["/bin/sh", "-c", "./exec_test.sh ${ITER}"]