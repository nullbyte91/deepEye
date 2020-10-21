# Custom Object Detector

## Dataset Download
We are using OIDv4_ToolKit to download a Open Image Dataset.

```bash
# Download a kit
git clone https://github.com/EscVM/OIDv4_ToolKit.git

cd OIDv4_ToolKit

# Install dep
pip3 install -r requirements.txt

# Downlaod a dataset
python3 main.py downloader --classes car --type_csv train
python3 main.py downloader --classes car --type_csv test
python3 main.py downloader --classes car --type_csv validation
```

## Setup Object Detection Model
```bash
python3 -m venv ODM
source ODM/bin/activate
pip3 install -r requirements.txt

# Object Detection Models
git clone --quiet https://github.com/tensorflow/models.git
cd models
git checkout 58d19c67e1d30d905dd5c6e5092348658fed80af
cd research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## TFRecord Preparation
```bash
# Convert txt to xml
python3 txt2xml.py

# Convert xml to csv
python3 xml2csv.py

# Convert csv to tfrecord
python3 csv2tfrecord.py
```

## Training 
```bash
# Modify the ssd_mobilenet_v2_coco.config
python3 model_main.py --pipeline_config_path=models/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config -num_train_steps=50 --num_eval_steps=10 --model_dir train_log
```

## Export a Trained Inference Graph
```bash
python3 export_inference_graph.py --input_type=image_tensor --pipeline_config_path=./models/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config --output_directory=./train_log --trained_checkpoint_prefix=model.ckpt-50.meta
```

## Running Inference: Checking what the trained model can detect
```bash
python3 tf_test.py
```

## Tensorflow to Openvino model conversion
```bash
sudo apt-get update && sudo apt-get install -y pciutils cpio

# Download OpenVino and setup
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz

chmod a+x install_GUI.sh
sudo bash install_GUI.sh
```

OpenVino Model conversion:
```bash
source /opt/intel/openvino/bin/setupvars.sh
```

SSD Changes:
```python
with open('ssd_v2_support.json', 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('"Postprocessor/ToFloat"', '"Postprocessor/Cast_1"')

# Write the file out again
with open('ssd_v2_support.json', 'w') as file:
  file.write(filedata)
```

```bash
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ./fine_tune_model/frozen_inference_graph.pb \
--tensorflow_use_custom_operations_config /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
--tensorflow_object_detection_api_pipeline_config ./fine_tune_model/pipeline.config \
--reverse_input_channels \
--data_type FP16 \
--output_dir ./openvino_out
```

OpenVino to depthAI blob conversion:
```bash
python3 openvino_depthai-blob.py -x ~/Desktop/OI_dataset/l_openvino_toolkit_p_2020.1.023/output/frozen_inference_graph.xml -b ~/Desktop/OI_dataset/l_openvino_toolkit_p_2020.1.023/output/frozen_inference_graph.bin -o output
```



