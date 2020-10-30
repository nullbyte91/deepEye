import depthai #core packages
import logging # Logger
import cv2     # computer vision package
import json    #json lib
from time import time, sleep, monotonic

from pathlib import Path #Path lib
from config import model, calib

# Object Tracker
from tracker import Tracker
from collision_avoidance import CrashAvoidance

log = logging.getLogger(__name__)

DEBUG = 1
class DepthAI:
    
    process_watchdog_timeout=10 #seconds

    def create_pipeline(self, config):
        self.device = depthai.Device('', False)
        log.info("Creating DepthAI pipeline...")

        self.pipeline = self.device.create_pipeline(config)
        if self.pipeline is None:
            raise RuntimeError("Pipeline was not created.")
        log.info("Pipeline created.")


    def reset_process_wd(self):
        global wd_cutoff
        wd_cutoff=monotonic()+self.process_watchdog_timeout
        return

    def __init__(self):
        self.threshold = 0.5

        self.config = config = {
            'streams': ['metaout', 'previewout'], 
            'depth': 
                {'calibration_file': str(Path(calib, 'depthai.calib').absolute()), 
                    'left_mesh_file': '', 
                    'right_mesh_file': '', 
                    'padding_factor': 0.3, 
                    'depth_limit_m': 10.0, 
                    'median_kernel_size': 7, 
                    'lr_check': False, 
                    'warp_rectify': 
                    {
                    'use_mesh': False, 
                    'mirror_frame': True, 
                    'edge_fill_color': 0
                    }
                }, 
            'ai': 
                {'blob_file': str(Path(model, 'mobilenet-ssd_v1.blob').absolute()), 
                'blob_file_config': str(Path(model, 'mobilenet-ssd_v1.json').absolute()), 
                'blob_file2': '', 
                'blob_file_config2': '', 
                'calc_dist_to_bb': True, 
                'keep_aspect_ratio': True, 
                'camera_input': 'rgb', 
                'shaves': 14, 
                'cmx_slices': 14, 
                'NN_engines': 1
                }, 
            'ot': 
                {'max_tracklets': 20, 
                'confidence_threshold': 0.5
                }, 
            'board_config': 
                {'swap_left_and_right_cameras': True, 
                'left_fov_deg': 71.86, 
                'rgb_fov_deg': 68.7938, 
                'left_to_right_distance_cm': 9.0, 
                'left_to_rgb_distance_cm': 2.0, 
                'store_to_eeprom': False, 
                'clear_eeprom': False, 
                'override_eeprom': False
                }, 
            'camera': 
                {'rgb': 
                    {'resolution_h': 1080, 
                    'fps': 30.0}, 
                    'mono': 
                    {'resolution_h': 720, 
                    'fps': 30.0
                    }
                }, 
            'app': 
                {'sync_video_meta_streams': False, 
                'sync_sequence_numbers': False, 
                'usb_chunk_KiB': 64
                }
            }
        # Create a pipeline config
        self.create_pipeline(self.config)

        blob_file = str(Path(model, 'mobilenet-ssd_depth.json'))

        with open(blob_file) as f:
            self.data = json.load(f)

        self.detection = []
        
        # Calculate collision_avoidance for bicycle
        # bus, car, dog,  horse, motorbike, train
        self.collision_avoidance = [2.0, 6.0, 7.0, 10.0, 12.0, 13.0, 14.0, 17.0]
        # Label map
        self.label = {
            0.0: "background",
            1.0: "aeroplane",
            2.0: "bicycle",
            3.0: "bird",
            4.0: "boat",
            5.0: "bottle",
            6.0: "bus", 
            7.0: "car",
            8.0: "cat",
            9.0: "chair",
            10.0: "cow",
            11.0: "diningtable",
            12.0: "dog",
            13.0: "horse",
            14.0: "motorbike",
            15.0: "person",
            16.0: "pottedplant",
            17.0: "sheep",
            18.0: "sofa",
            19.0: "train",
            20.0: "tvmonitor",
        }

        self.reset_process_wd()

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def capture(self):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  

        # Label mapping
        try:
            labels = self.data['mappings']['labels']
        except:
            labels = None
            print("Labels not found in json!")

        while True:
            #print("debug")
            # retreive data from the device
            # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
            nnet_packets, data_packets = self.pipeline.get_available_nnet_and_data_packets()
           
            packets_len = len(nnet_packets) + len(data_packets)
            if packets_len != 0:
                self.reset_process_wd()
            else:
                cur_time=monotonic()
                if cur_time > wd_cutoff:
                    print("process watchdog timeout")
                    os._exit(10)
            
            # Get the detection 
            for _, nnet_packet in enumerate(nnet_packets):
                self.detection  = []
                detections = nnet_packet.getDetectedObjects()
                for detection in detections:
                    detection_dict = detection.get_dict()
                    if detection_dict['confidence'] > 0.5:
                        self.detection.append(detection_dict)
            
            # Post processing
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
                    
                    frame_o = frame.copy()
                    for e in self.detection:
                        color = (0, 255, 0) # bgr
                        label = e['label']
                        if label in self.collision_avoidance:
                            # Create dic for tracking
                            boxes.append({
                            'detector': "MobileNet SSD",
                            'label': self.label[label],
                            'conf': e['confidence'],
                            'left': int(e['x_min'] * img_w),
                            'top': int(e['y_min'] * img_h),
                            'right': int(e['x_max'] * img_w),
                            'bottom': int(e['y_max'] * img_h),
                            'distance_x': e['depth_x'],
                            'distance_y': e['depth_y'],
                            'distance_z': e['depth_z'],
                            })
                            color = (0, 0, 255) # bgr
                            
                        pt1 = int(e['x_min']  * img_w), int(e['y_min']    * img_h)
                        pt2 = int(e['x_max'] * img_w), int(e['y_max'] * img_h)

                        x1, y1 = pt1
                        x2, y2 = pt2

                        cv2.rectangle(frame, pt1, pt2, color)

                        pt_t1 = x1, y1 + 20
                        cv2.putText(frame, labels[int(e['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        pt_t2 = x1, y1 + 40
                        cv2.putText(frame, '{:.2f}'.format(100*e['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                        if DEBUG:
                            if self.config['ai']['calc_dist_to_bb']:
                                pt_t3 = x1, y1 + 60
                                cv2.putText(frame, 'x1:' '{:7.3f}'.format(e['depth_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                                pt_t4 = x1, y1 + 80
                                cv2.putText(frame, 'y1:' '{:7.3f}'.format(e['depth_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                                pt_t5 = x1, y1 + 100
                                cv2.putText(frame, 'z1:' '{:7.3f}'.format(e['depth_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                        
                        # frame = self.image_resize(frame, 500)

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
        # Pass class along with Z and X coordinates
        pts_l = [(item['distance_x'], item['distance_z'], item['label']) for item in results]

        # Pass the points to tracker
        tracker_objs, obj_class = tracker.update(pts_l, log)

        #Pass the tracker objects to collision_avoidance
        crash_alert = crash_avoidance.parse(tracker_objs, obj_class)
        
if __name__ == "__main__":
    main()