import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from numpy import clip
from utils.ie_module import Module

class SemanticSegmentation(Module):
    def __init__(self, model):
        super(SemanticSegmentation, self).__init__(model)

        self.input_blob = next(iter(model.inputs))
        self.input_name = next(iter(model.input_info))

        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape

        print("Input Shape: {}".format(self.input_shape))
        print("Output Shape: {}".format(self.output_shape))
    
    def resize_input(self, frame, target_shape):
        assert len(frame.shape) == len(target_shape), \
            "Expected a frame with %s dimensions, but got %s" % \
            (len(target_shape), len(frame.shape))

        assert frame.shape[0] == 1, "Only batch size 1 is supported"
        n, c, h, w = target_shape

        input = frame[0]
        if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
            input = input.transpose((1, 2, 0)) # to HWC
            input = cv2.resize(input, (w, h))
            input = input.transpose((2, 0, 1)) # to CHW

        return input.reshape((n, c, h, w))

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = self.resize_input(frame, self.input_shape)
        return input

    def draw_masks(self, result, width, height):
        '''
        Draw semantic mask classes onto the frame.
        '''
        # Create a mask with color by class
        classes = cv2.resize(result[0].transpose((1, 2, 0)), (width, height),
                            interpolation=cv2.INTER_NEAREST)
        unique_classes = np.unique(classes)
        out_mask = classes * (255 / 20)

        # Stack the mask so FFmpeg understands it
        out_mask = np.dstack((out_mask, out_mask, out_mask))
        out_mask = np.uint8(out_mask)

        return out_mask, unique_classes

    def post_processing(self, frame):
        print(frame.shape)
        H, W, C = frame.shape
        seg_image = Image.open("../../../demo/009649.png")
        palette = seg_image.getpalette() # Get a color palette        
        outputs = self.get_outputs()[0][self.output_blob]
        print(outputs.shape)
        output = outputs[0][0]
        print(output.shape)
        print(type(output))
        output  = np.array(output, dtype='uint8')
        print(type(output))
        print("Height: {}".format(H))
        print("Weight: {}".format(W))
        output = cv2.resize(output, (W, H))

        image = Image.fromarray(np.uint8(output), mode="P")
        image.putpalette(palette)
        image = image.convert("RGB")
        
        image = np.asarray(image)
        print("Frame size: {}".format(frame.shape))
        print("image size: {}".format(image.shape))
        image = cv2.addWeighted(frame, 1, image, 0.9, 0)
        # plt.imshow(image)
        # plt.pause(0.001)
        #plt.show()     
        # plt.show()
        cv2.imshow("Result", image)
        cv2.waitKey(1)
        
    def start_async(self, frame):
        input = self.preprocess(frame)
        input_data = {self.input_name: input}
        self.enqueue(input_data)

