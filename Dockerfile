# Use an official Python runtime as a parent image
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM liuyuweitarek/pytorch:2.4.0-py3.11-cuda12.5.1-ubuntu22.04

# Set the working directory to /app
WORKDIR /ws

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --no-install-recommends \ 
    && apt-get install -y apt-utils 
# RUN apt install git python3.8 python3.8-dev python3.8-venv python3-pip python3-tk python-is-python3 -y && rm -rf /var/lib/apt/lists/*
RUN apt install git python3-pip python3-tk python-is-python3 -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update --no-install-recommends \ 
    && apt-get install -y apt-utils 

RUN apt-get install -y \
  build-essential \
  cmake \
  cppcheck \
  gdb \
  git \
  lsb-release \
  software-properties-common \
  sudo \
  vim \
  wget \
  tmux \
  curl \
  less \
  net-tools \
  byobu \
  libgl-dev \
  iputils-ping \
  nano \
  unzip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Add a user with the same user_id as the user outside the container
# Requires a docker build argument `user_id`
ARG user_id=1000
ENV USERNAME developer
RUN useradd -U --uid ${user_id} -ms /bin/bash $USERNAME \
 && echo "$USERNAME:$USERNAME" | chpasswd \
 && adduser $USERNAME sudo \
 && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

# Commands below run as the developer user
USER $USERNAME

# When running a container start in the developer's home folder
WORKDIR /home/$USERNAME

RUN pip3 install --upgrade pip

#ros2
RUN export DEBIAN_FRONTEND=noninteractive \
 && sudo apt-get update \
 && sudo apt-get install -y \
   tzdata \
 && sudo ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
 && sudo dpkg-reconfigure --frontend noninteractive tzdata \
 && sudo apt-get clean 

RUN sudo apt update && sudo apt install locales
RUN sudo locale-gen en_US en_US.UTF-8
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

RUN sudo apt-get install software-properties-common -y
RUN sudo apt-get install --reinstall python3-apt -y
#RUN sudo add-apt-repository universe
RUN sudo apt update && sudo apt install curl -y
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN sudo apt update
RUN sudo apt upgrade -y
RUN sudo apt install ros-humble-desktop -y
RUN sudo apt install ros-dev-tools -y
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc


RUN pip3 install uv
