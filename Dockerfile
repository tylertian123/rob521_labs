FROM osrf/ros:noetic-desktop-full

ENV ROS_DISTRO=noetic
ENV DEBIAN_FRONTEND=noninteractive
ENV DISABLE_ROS1_EOL_WARNINGS=1

WORKDIR /tmp/rob521_docker_build

# Basic packages
RUN apt update && apt install -y \
    bash fish git wget curl python3-pip build-essential \
    ros-${ROS_DISTRO}-catkin python3-catkin-tools python3-rosdep \
    python3-rosinstall python3-rosinstall-generator

# Create user that the container runs as
# Make sure to add to the relevant groups in order to grant access to hardware
RUN useradd -ms /bin/bash rob521 && usermod -aG dialout,video,tty,sudo rob521 \
    && echo 'rob521 ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER rob521

# Install bash compatibility layer for fish shell
# Run this as user rob521
RUN wget https://raw.githubusercontent.com/oh-my-fish/oh-my-fish/master/bin/install -O /tmp/omf-install \
    && /usr/bin/fish /tmp/omf-install \
    && /usr/bin/fish -c 'omf install bass'

# Switch back to root for the rest of the installation
USER root

# Turtlebot specific packages and setup
ENV TURTLEBOT3_MODEL=waffle_pi
RUN apt install -y ros-noetic-joy ros-noetic-teleop-twist-joy \
    ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
    ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
    ros-noetic-rosserial-python ros-noetic-rosserial-client \
    ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
    ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
    ros-noetic-compressed-image-transport ros-noetic-rqt* \
    ros-noetic-rviz ros-noetic-gmapping \
    ros-noetic-navigation ros-noetic-interactive-markers \
    ros-noetic-dynamixel-sdk ros-noetic-turtlebot3-msgs ros-noetic-turtlebot3
RUN python3 -m pip install --upgrade scikit-image pygame tqdm

# Get a temp copy of the repo to use rosdep to install all deps
COPY . ./repo_deps_tmp
RUN rosdep install -y --from-paths ./repo_deps_tmp/src --ignore-src --rosdistro ${ROS_DISTRO}

RUN rm -rf /tmp/rob521_docker_build

# Setup bashrc and config.fish for the correct environment
RUN echo "source /home/rob521/rob521_labs/devel/setup.bash" >> /home/rob521/.bashrc \
    && echo "source /opt/ros/noetic/share/rosbash/rosfish" >> /home/rob521/.config/fish/config.fish
    && echo "bass source /home/rob521/rob521_labs/devel/setup.bash" >> /home/rob521/.config/fish/config.fish

# Switch to correct user
USER rob521
