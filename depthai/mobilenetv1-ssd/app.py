import depthai #depthai core utils
import logging as log
import os
import json
import cv2

from argparse import ArgumentParser
from time import time, sleep, monotonic
# Specific to mobilenetv1-ssd - Not sure v2 
from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd

# Glob var but revist them 
calc_dist_to_bb = True
calib_fpath = "/home/nullbyte/Desktop/vision-system/depthai/mobilenetv1-ssd/resources/depthai.calib"
depth_ai_cmd_file = "/home/nullbyte/Desktop/depthai/depthai.cmd"

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    return parser

def main():

    # Depth AI version
    print("DepthAI Version: {}".format(depthai.__version__))
    
    log.basicConfig(level=log.INFO)
    # Grab command line args
    args = build_argparser().parse_args()
    
    decode_nn=decode_mobilenet_ssd
    show_nn=show_mobilenet_ssd
    
    # Fetch blob and json path details
    cnn_model_path = os.path.abspath(args.model)
    blob_file_path = cnn_model_path + ".blob"
    suffix=""
    if calc_dist_to_bb:
        suffix="_depth"
    blob_file_config_path = cnn_model_path + suffix + ".json"

    # check the file existance
    if not os.path.isfile(blob_file_path):
        log.error("NN blob not found in {}".format(blob_file_path))
        exit(-1)

    if not os.path.isfile(blob_file_config_path):
        log.error("NN config not found in {}".format(blob_file_config_path))
        exit(-1)
    
    # Open a NN model config
    with open(blob_file_config_path) as f:
        data = json.load(f)

    # Check the label info from the json
    try:
        labels = data['mappings']['labels']
        log.info("Label Found {}".format(labels))
    except:
        labels = None
        print("Labels not found in json!")
    
    # Note: We have to compile a model if run multiple model

    #default
    # Why we are using only 7 SHAVES why not 14
    shave_nr = 7
    cmx_slices = 7
    NCE_nr = 1

    # Dep to generate the config file
    # 1. calibration file

    config = {
    # Possible streams:
    # ['left', 'right','previewout', 'metaout', 'depth_raw', 'disparity', 'disparity_color']
    # If "left" is used, it must be in the first position.
    # To test depth use:
    # 'streams': [{'name': 'depth_raw', "max_fps": 12.0}, {'name': 'previewout', "max_fps": 12.0}, ],
    'streams': ['metaout', 'previewout'], #default
    'depth':
    {
        'calibration_file': calib_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
    },
    'ai':
    {
        'blob_file': blob_file_path,
        'blob_file_config': blob_file_config_path,
        'blob_file2': "",
        'blob_file_config2': "",
        'calc_dist_to_bb': False, # ?????
        'keep_aspect_ratio': False, #???
        'camera_input': "rgb", # can we assign left and right
        'shaves' : shave_nr,
        'cmx_slices' : cmx_slices,
        'NN_engines' : NCE_nr,
    },
    # object tracker
    'ot':
    {
        'max_tracklets'        : 20, #maximum 20 is supported
        'confidence_threshold' : 0.5, #object is tracked only for detections over this threshold
    },
    'board_config':
    {
        'swap_left_and_right_cameras': True, # True for 1097 (RPi Compute) and 1098OBC (USB w/onboard cameras)
        'left_fov_deg': 71.86, # Same on 1097 and 1098OBC
        'rgb_fov_deg': 68.7938,
        'left_to_right_distance_cm': 9.0, # Distance between stereo cameras
        'left_to_rgb_distance_cm': 2.0, # Currently unused
        'store_to_eeprom': False,
        'clear_eeprom': False,
        'override_eeprom': False,
    },
    'camera':
    {
        'rgb':
        {
            # 3840x2160, 1920x1080
            # only UHD/1080p/30 fps supported for now
            'resolution_h': 1080,
            'fps': 30.0,
        },
        'mono':
        {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            'resolution_h': 720, # what are the other or how to decide?
            'fps': 30.0,
        },
    },
    'app':
    {
        'sync_video_meta_streams': False, # How to use sync video meta stream ?
    },
    #'video_config':
    #{
    #    'rateCtrlMode': 'cbr',
    #    'profile': 'h265_main', # Options: 'h264_baseline' / 'h264_main' / 'h264_high' / 'h265_main'
    #    'bitrate': 8000000, # When using CBR
    #    'maxBitrate': 8000000, # When using CBR
    #    'keyframeFrequency': 30,
    #    'numBFrames': 0,
    #    'quality': 80 # (0 - 100%) When using VBR
    #}
    }

    stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in config['streams']]

    
    # Initialize the DepthAI Device
    if not depthai.init_device(depth_ai_cmd_file):
        print("Error initializing device. Try to reset it.")
        exit(1)
    
    log.info('Available streams: ' + str(depthai.get_available_steams()))

    # create the pipeline, here is the first connection with the device
    p = depthai.create_pipeline(config=config)

    if p is None:
        print('Pipeline is not created.')
        exit(3)
    
    # Check this for reference
    # https://github.com/luxonis/depthai-core/blob/e1738435ca239fee3c435e9eddadc8b8aff81e6b/src/device.cpp
    nn2depth = depthai.get_nn_to_depth_bbox_mapping()


    # Local var
    t_start = time()
    frame_count = {}
    frame_count_prev = {}
    nnet_prev = {}
    nnet_prev["entries_prev"] = {}
    nnet_prev["nnet_source"] = {}
    frame_count['nn'] = {}
    frame_count_prev['nn'] = {}

    NN_cams = {'rgb', 'left', 'right'}

    for cam in NN_cams:
        nnet_prev["entries_prev"][cam] = []
        nnet_prev["nnet_source"][cam] = []
        frame_count['nn'][cam] = 0
        frame_count_prev['nn'][cam] = 0
    
    stream_windows = []
    for s in stream_names:
        if s == 'previewout':
            for cam in NN_cams:
                stream_windows.append(s + '-' + cam)
        else:
            stream_windows.append(s)

    for w in stream_windows:
        frame_count[w] = 0
        frame_count_prev[w] = 0

    
    # Why we are doing watchdog reset?
    process_watchdog_timeout=10 #seconds
    def reset_process_wd():
        global wd_cutoff
        wd_cutoff=monotonic()+process_watchdog_timeout
        return

    reset_process_wd()

    while True:
        # retreive data from the device
        # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
        nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

        packets_len = len(nnet_packets) + len(data_packets)

        #if packet len 0 the reset a watchdog timer
        if packets_len != 0:
            reset_process_wd()
        else:
            cur_time=monotonic()
            if cur_time > wd_cutoff:
                print("process watchdog timeout")
                os._exit(10)

        for _, nnet_packet in enumerate(nnet_packets):
            camera = nnet_packet.getMetadata().getCameraName()
            nnet_prev["nnet_source"][camera] = nnet_packet #object pointer address
            nnet_prev["entries_prev"][camera] = decode_nn(nnet_packet, config=config) # Need to understand how they are decoding it
            frame_count['metaout'] += 1
            frame_count['nn'][camera] += 1

        for packet in data_packets:
            window_name = packet.stream_name
            if packet.stream_name not in stream_names:
                continue # skip streams that were automatically added
            packetData = packet.getData()
            if packetData is None:
                print('Invalid packet data!')
                continue
            elif packet.stream_name == 'previewout':
                camera = packet.getMetadata().getCameraName()
                window_name = 'previewout-' + camera
                # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
                # change shape (3, 300, 300) -> (300, 300, 3)
                data0 = packetData[0,:,:]
                data1 = packetData[1,:,:]
                data2 = packetData[2,:,:]
                frame = cv2.merge([data0, data1, data2])

                # The frame and decode obj passed to post processing
                nn_frame = show_nn(nnet_prev["entries_prev"][camera], frame, labels=labels, config=config)
                
                cv2.putText(nn_frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                cv2.putText(nn_frame, "NN fps: " + str(frame_count_prev['nn'][camera]), (2, frame.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
                cv2.imshow(window_name, nn_frame)

        t_curr = time()
        if t_start + 1.0 < t_curr:
            t_start = t_curr
            # print("metaout fps: " + str(frame_count_prev["metaout"]))

            stream_windows = []
            for s in stream_names:
                if s == 'previewout':
                    for cam in NN_cams:
                        stream_windows.append(s + '-' + cam)
                        frame_count_prev['nn'][cam] = frame_count['nn'][cam]
                        frame_count['nn'][cam] = 0
                else:
                    stream_windows.append(s)
            for w in stream_windows:
                frame_count_prev[w] = frame_count[w]
                frame_count[w] = 0
    
        key = cv2.waitKey(1)
        if key == ord('c'):
            depthai.request_jpeg()
        elif key == ord('f'):
            depthai.request_af_trigger()
        elif key == ord('1'):
            depthai.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO)
        elif key == ord('2'):
            depthai.request_af_mode(depthai.AutofocusMode.AF_MODE_CONTINUOUS_VIDEO)
        elif key == ord('q'):
            break
    
if __name__ == "__main__":
    main()