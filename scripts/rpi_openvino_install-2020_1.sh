#!/bin/bash

# A script to install OpenVino toolchain 2020 1 version

#Main starts from here
# Downlaod openvino tool chain
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.1/l_openvino_toolkit_runtime_raspbian_p_2020.1.023.tgz

# Create openvino install path
sudo mkdir -p /opt/intel/openvino

# Unzip the toolchain
sudo tar -xvf l_openvino_toolkit_runtime_raspbian_p_2020.1.023.tgz --strip 1 -C /opt/intel/openvino

# Install cmake 
sudo apt install cmake

# Export a path
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc

# Add use for USB
sudo usermod -a -G users "$(whoami)"
