#!/bin/bash
# A script to build inference engine for ARM arch - Tested on Jetson Nano - 4.3 Jetpack

# System update and Dep Install
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential 

# Update cmake from source
cd ~/
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4.tar.gz
tar xvzf cmake-3.14.4.tar.gz
cd ~/cmake-3.14.4

## Configure and install
./bootstrap
make -j4
sudo make install

# Install OpenCV
cd ~/
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake –DCMAKE_BUILD_TYPE=Release –DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install

# Install OpenVino Inference Engine
cd ~/
git clone https://github.com/openvinotoolkit/openvino.git

cd ~/openvino/inference-engine
git submodule update --init --recursive

## Dep install 
cd ~/openvino
sh ./install_dependencies.sh

## Build
export OpenCV_DIR=/usr/local/lib

cd ~/openvino
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MKL_DNN=OFF \
    -DENABLE_CLDNN=ON \
    -DENABLE_GNA=OFF \
    -DENABLE_SSE42=OFF \
    -DTHREADING=SEQ \
    -DENABLE_SAMPLES=ON \
    -DENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=`which python3.6` \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.6
    ..
make -j4

# Update env
echo "export PYTHONPATH=$PYTHONPATH:~/openvino/bin/aarch64/Release/lib/python_api/python3.6/" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/openvino/bin/aarch64/Release/lib/" >> ~/.bashrc