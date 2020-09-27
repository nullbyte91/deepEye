import depthai #core packages
import logging # Logger
import cv2     # computer vision package
import json    #json lib

from pathlib import Path #Path lib
from config import model

# Object Tracker
from tracker import Tracker
from crash_avoidance import CrashAvoidance

log = logging.getLogger(__name__)

DEBUG = 0
class DepthAI:
    def create_pipeline(self, config):
        self.device = depthai.Device('', False)
        log.info("Creating DepthAI pipeline...")

        self.pipeline = self.device.create_pipeline(config)
        if self.pipeline is None:
            raise RuntimeError("Pipeline was not created.")
        log.info("Pipeline created.")

    def __init__(self):
        self.threshold = 0.5
        self.config = {
            'streams': ['previewout', 'metaout'],
            "depth": {
                "calibration_file": "",
                "padding_factor": 0.3,
                "depth_limit_m": 10.0
            },
            'ai': {
                "calc_dist_to_bb": True,
                "keep_aspect_ratio": True,
                'blob_file': str(Path(model, 'mobilenet-ssd.blob').absolute()),
                'blob_file_config': str(Path(model, 'mobilenet-ssd_depth.json').absolute())
            },
            'camera':
            {
                'rgb':
                {
                    # 3840x2160, 1920x1080
                    # only UHD/1080p/30 fps supported for now
                    'resolution_h': 1080,
                    'fps': 30,
                },
                'mono':
                {
                    # 1280x720, 1280x800, 640x400 (binning enabled)
                    'resolution_h': 720,
                    'fps': 30,
                },
            },

            }

        # Create a pipeline config
        self.create_pipeline(self.config)

        blob_file = str(Path(model, 'mobilenet-ssd_depth.json'))

        with open(blob_file) as f:
            self.data = json.load(f)

        self.detection = []
        
        # Calculate colliusion_avoidance for bicycle
        # bus, car, dog,  horse, motorbike, train
        self.colliusion_avoidance = [2.0, 6.0, 7.0, 10.0, 12.0, 13.0, 14.0, 17.0]

    def capture(self):

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  

        # Label mapping
        try:
            labels = self.data['mappings']['labels']
        except:
            labels = None
            print("Labels not found in json!")

        while True:
            # retreive data from the device
            # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
            nnet_packets, data_packets = self.pipeline.get_available_nnet_and_data_packets()
            for _, nnet_packet in enumerate(nnet_packets):
                self.detection  = []

                # Shape: [1, 1, N, 7], where N is the number of detected bounding boxes
                for _, e in enumerate(nnet_packet.entries()):
                    # for MobileSSD entries are sorted by confidence
                    # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
                    if e[0]['image_id'] == -1.0 or e[0]['conf'] == 0.0:
                        break
                    # save entry for further usage (as image package may arrive not the same time as nnet package)
                    # the lower confidence threshold - the more we get false positives
                    #print( e[0]['conf'])
                    if e[0]['conf'] > 0.5:
                        self.detection.append(e)
            
            boxes = []
            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    if data is None:
                        continue
                    # The format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
                    # change shape (3, 300, 300) -> (300, 300, 3).
                    data0 = data[0, :, :]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    frame = cv2.merge([data0, data1, data2])

                    img_h = frame.shape[0]
                    img_w = frame.shape[1]

                    for e in self.detection:
                        color = (0, 255, 0) # bgr
                        label = e[0]['label']
                        if label in self.colliusion_avoidance:
                            # Create dic for tracking
                            boxes.append({
                            'detector': "MobileNet SSD",
                            'conf': e[0]['conf'],
                            'left': int(e[0]['x_min'] * img_w),
                            'top': int(e[0]['y_min'] * img_h),
                            'right': int(e[0]['x_max'] * img_w),
                            'bottom': int(e[0]['y_max'] * img_h),
                            'distance_x': e[0]['distance_x'],
                            'distance_y': e[0]['distance_y'],
                            'distance_z': e[0]['distance_z'],
                            })
                            color = (0, 0, 255) # bgr
                            
                        frame_o = frame.copy()

                        pt1 = int(e[0]['x_min']  * img_w), int(e[0]['y_min']    * img_h)
                        pt2 = int(e[0]['x_max'] * img_w), int(e[0]['y_max'] * img_h)


                        x1, y1 = pt1
                        x2, y2 = pt2

                        cv2.rectangle(frame, pt1, pt2, color)

                        pt_t1 = x1, y1 + 20
                        cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        pt_t2 = x1, y1 + 40
                        cv2.putText(frame, '{:.2f}'.format(100*e[0]['conf']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                        if DEBUG:
                            if self.config['ai']['calc_dist_to_bb']:
                                pt_t3 = x1, y1 + 60
                                cv2.putText(frame, 'x1:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                                pt_t4 = x1, y1 + 80
                                cv2.putText(frame, 'y1:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                                pt_t5 = x1, y1 + 100
                                cv2.putText(frame, 'z1:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                        
                        cv2.imshow("output", frame)
                        if cv2.waitKey(1) == ord('q'):
                            cv2.destroyAllWindows()
                            exit(0)

                        yield frame_o, boxes

def main():
    # depth AI
    di = DepthAI()
    # Tracker
    tracker = Tracker(log)
    # Cross avoidance
    crash_avoidance = CrashAvoidance()
    
    for frame, results in di.capture():
        pts = [(item['distance_x'], item['distance_z']) for item in results]
        tracker_objs = tracker.update(pts, log)
        crash_alert = crash_avoidance.parse(tracker_objs)
        
if __name__ == "__main__":
    main()