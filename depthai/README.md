# Depth AI Tasks

| Task  |  Status | Notes|
|---|---|---
| mobilenet-SSD v1 | Done | Used default depth AI configuration
| mobilenet-SSD v2  |   |
| tracking   |   |
| depth information  |   |
| Yolo |   |
| Calibration for test setup  |   |
| Benchmark and analysis  |   |
|   |   |
|   |   |

## Tensorflow to into Intel Movidius binary format
### Level 1 conversion
```bash
# ssd_mobilenet_v2_coco
python3 mo_tf.py --input_model=/home/nullbyte/Desktop/openvino_models/model/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --transformations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/nullbyte/Desktop/openvino_models/model/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels -o ~/Desktop/mo
_models/mobilenet-ssd-v2/
```

### Level 2 conversion
```bash
# ssd_mobilenet_v2_coco - FP32 
 ./bin/aarch64/Release/myriad_compile -m ~/Desktop/mo_models/mobilenet-ssd-v2/frozen_inference_graph.xml -o  ~/Desktop/mo_models/
mobilenet-ssd-v2/FP32/mobilenet-ssd-v2.blob -ip FP32 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4
```

---
**NOTE**

We need to build a json file based on Output blob dimension and the classes.
---