import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log

from argparse import ArgumentParser
from inference import Network

from imutils.video import FPS

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-l", "--cpu_extension",
                    help="Optional. Required for CPU custom layers. "
                        "Absolute path to a shared library with the kernels implementations.",
                    type=str, default=None)
    return parser


def preprocessing(frame, in_w, in_h):        
    image_resize = cv2.resize(frame, (in_w, in_h), interpolation = cv2.INTER_AREA)
    image = np.moveaxis(image_resize, -1, 0)

    return image 
def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    # Initialize the Inference Engine
    infer_network = Network()

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args, args.device, num_requests=0)

    # Get a Input blob shape
    _, _, in_h, in_w = infer_network.get_input_shape()

    
    # Get a output blob name
    output_name = infer_network.get_output_name()

    # Handle the input stream
    try:
        cap = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    cap.open(args.input)
    _, frame = cap.read()
    
    fh = frame.shape[0]
    fw = frame.shape[1]
    
    fps = FPS().start()

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(1)

        image = preprocessing(frame, in_w, in_h)

        # Perform inference on the frame
        infer_network.exec_net(image, request_id=0)
        
        # Get the output of inference
        if infer_network.wait(request_id=0) == 0:
            result = infer_network.get_output(request_id=0)
            print(result.shape)
            
            # Analyze road segmentation results
            result = np.squeeze(result, 0)
            result  =  result . transpose ( 1 , 2 , 0 )   # HWC
            result = np.argmax(result, 2)
            
            hh, ww = result.shape
            mask = np.zeros((hh, ww, 3), dtype=np.uint8)
            mask[np.where(result > 0)] = (0, 255, 255)  # yellow
            mask[np.where(result > 1)] = (255, 0, 255)  #
            
            #cv2.imshow("mask", mask)
            
            # Superimpose output results
            mask = cv2.resize(mask, dsize=(frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(mask, 0.2, frame, 0.8, 0)
        

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Break if escape key pressed
        if key_pressed == 27:
            break
        
        fps.update()
    
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()

    cv2.destroyAllWindows()
    
    fps.stop()

    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Set log to INFO
    log.basicConfig(level=log.CRITICAL)

    # Grab command line args
    args = build_argparser().parse_args()
    
    # Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()