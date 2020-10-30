# Model Analysis

> Hardware Configuration
> CPU : i7-8665U
> VPU : myriad x NCS2

## Intel Models
|Model Name   |Complexity - GFLOPs   | Size - MP  | CPU FPS - FP32 |  Jetson + VPU FPS - FP16 | RPI + VPU FPS - FP16
|---          |---                   |---         |---             |---                       |---
|pedestrian-detection-adas-0002      | 2.836   |1.165   |28   |6.86   |4.53
|pedestrian-detection-adas-binary-0001   |0.945   |1.165|33   |Not Supported Layer Found   |Not Supported Layer Found
|pedestrian-and-vehicle-detector-adas-0001   |	3.974   |1.650	   |24.09   |6.18   |4.37
|vehicle-detection-adas-0002   |2.798   |1.079   |26   |7.51   |4.85
|vehicle-detection-adas-binary-0001   |0.942   |1.079   |28   |Not Supported Layer Found   |Not Supported Layer Found
|road-segmentation-adas-0001   |4.770   |4.770   |8.48   |1.94   |1.02
|semantic-segmentation-adas-0001   |58.572   |6.686   |2.06  |Memory Error   |Memory Error

## Public Models
|Model Name   |Complexity - GFLOPs   | Size - MP  | CPU FPS - FP32 |  Jetson + VPU FPS - FP16 | RPI + VPU FPS - FP16
|---          |---                   |---         |---             |---                       |---
|YOLO v3      | 65.9843   |237   |1.86   |1.4   |Crash |
|SSD with MobileNet |2.316~2.494   |65 |22   |10.4   |6.51

## Depth AI HW Model analysis
| Model  | SAHVES 4 - FPS | SAHVES 7 - FPS  | GFlops  |  mParams |   |
|---|---|---|---|---|---|
| ssd_mobilenet_v1_coco  | 28-30  |   | 2.316~2.494  | 5.783~6.807  |   |
| ssd_mobilenet_v2_coco  | 12-13  |  14-15 | 3.775  | 16.818   |   |
|   |   |   |   |   |   |
|   |   |   |   |   |   |