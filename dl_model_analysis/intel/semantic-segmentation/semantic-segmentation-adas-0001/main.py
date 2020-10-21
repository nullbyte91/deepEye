import logging as log
import sys
import time
import os.path as osp
import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from argparse import ArgumentParser

from utils.ie_module import InferenceContext
from core.semantic_detection import SemanticSegmentation

#Global 
DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """

    parser = ArgumentParser()
    parser.add_argument("-input", "--input", required=True, type=str,
                       help="Path to image or video file or enter cam for webcam")
    parser.add_argument("-model", "--detection_model", required=True, type=str,
                        help="Path to an .xml file with a trained Face Detection model") 
    parser.add_argument('-device', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "detection model (default: %(default)s)")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    parser.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    parser.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    parser.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    return parser

class ProcessOnFrame:
    # Queue size will be used to put frames in a queue for
    # Inference Engine
    QUEUE_SIZE = 10

    def __init__(self, args):
        used_devices = set([args.device])

        # Create a Inference Engine Context
        self.context = InferenceContext()
        context = self.context

        # Load OpenVino Plugin based on device selection
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        start_time = time.perf_counter()

        # Load face detection model on Inference Engine
        segmentation = self.load_model(args.detection_model)

        stop_time = time.perf_counter()
        print("[INFO] Model Load Time: {}".format(stop_time - start_time))

        self.segmentation = SemanticSegmentation(segmentation)

        self.segmentation.deploy(args.device, context)
        
        log.info("Models are loaded")

    def load_model(self, model_path):
        """
        Initializing IENetwork(Inference Enginer) object from IR files:
        
        Args:
        Model path - This should contain both .xml and .bin file
        :return Instance of IENetwork class
        """
        
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        
        print(model_weights_path)
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
           
        # Load model on IE
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
         
        return model
    
    def frame_pre_process(self, frame):
        """
        Pre-Process the input frame given to model
        Args:
        frame: Input frame from video stream
        Return:
        frame: Pre-Processed frame
        """
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)
        return frame

    def detection(self, frame):
        image = frame.copy()
        frame = self.frame_pre_process(frame)

        self.segmentation.clear()

        self.segmentation.start_async(frame)
        self.segmentation.post_processing(image)


class VisionSystem:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self, args):
        self.frame_processor = ProcessOnFrame(args)
        self.print_perf_stats = args.perf_stats

    def process(self):
        input_stream = cv2.VideoCapture("../../../demo/Paris.mp4")
        
        frame_count = 0
            
        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break
            
            frame_count+=1
            print("Frame cout: {}".format(frame_count))

            self.frame_processor.detection(frame)
    def run(self, args):
        if args.input == "cam":
            path = "0"
        else:
            path = args.input

        # input_stream = VisionSystem.open_input_stream(path)
        
        # if input_stream is None or not input_stream.isOpened():
        #     log.error("Cannot open input stream: %s" % args.input)

        # # FPS init
        # fps = input_stream.get(cv2.CAP_PROP_FPS)

        # # Get the Frame org size
        # frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #               int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # # Get the frame count if its a video
        # self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        self.process()

    @staticmethod
    def open_input_stream(path):
        """
        Open the input stream
        """
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

def main():
    args = build_argparser().parse_args()
    
    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)
    detection = VisionSystem(args)
    detection.run(args)

if __name__ == "__main__":
    main()