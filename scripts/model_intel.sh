#!/bin/bash
# Use this only when you have issue with Windows download env
# Otherwise use download.py from openvino toolkit

dModel="./model/"

# Main starts here
mkdir -p ${dModel}
# pedestrian-detection-adas-0002
pushd ${dModel}
mkdir -p pedestrian-detection-adas-0002
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.xml -P \
    pedestrian-detection-adas-0002
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.bin -P \
    pedestrian-detection-adas-0002

# pedestrian-detection-adas-binary-0001
mkdir -p pedestrian-detection-adas-binary-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/pedestrian-detection-adas-binary-0001/FP32-INT1/pedestrian-detection-adas-binary-0001.xml -P \
    pedestrian-detection-adas-binary-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/pedestrian-detection-adas-binary-0001/FP32-INT1/pedestrian-detection-adas-binary-0001.bin -P \
    pedestrian-detection-adas-binary-0001

# pedestrian-and-vehicle-detector-adas-0001
mkdir -p pedestrian-and-vehicle-detector-adas-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/pedestrian-and-vehicle-detector-adas-0001/FP32/pedestrian-and-vehicle-detector-adas-0001.xml -P \
    pedestrian-and-vehicle-detector-adas-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/pedestrian-and-vehicle-detector-adas-0001/FP32/pedestrian-and-vehicle-detector-adas-0001.bin -P \
    pedestrian-and-vehicle-detector-adas-0001

# vehicle-detection-adas-0002
mkdir -p vehicle-detection-adas-0002
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.xml -P \
vehicle-detection-adas-0002
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin -P \
vehicle-detection-adas-0002

# vehicle-detection-adas-binary-0001
mkdir -p vehicle-detection-adas-binary-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/vehicle-detection-adas-binary-0001/FP32-INT1/vehicle-detection-adas-binary-0001.xml -P \
vehicle-detection-adas-binary-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/vehicle-detection-adas-binary-0001/FP32-INT1/vehicle-detection-adas-binary-0001.bin -P \
vehicle-detection-adas-binary-0001

# road-segmentation-adas-0001
mkdir -p road-segmentation-adas-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml -P \
road-segmentation-adas-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin -P \
road-segmentation-adas-0001

# semantic-segmentation-adas-0001
mkdir -p semantic-segmentation-adas-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/semantic-segmentation-adas-0001/FP32/semantic-segmentation-adas-0001.xml -P \
semantic-segmentation-adas-0001
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/semantic-segmentation-adas-0001/FP32/semantic-segmentation-adas-0001.bin -P \
semantic-segmentation-adas-0001

pushd 