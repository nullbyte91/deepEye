import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.ie = None
        self.network = None
        self.input_blob = None
        self.output_blob = None 
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, args, device="CPU", num_requests=0):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        
        # Initialize the plugin
        self.ie = IECore()

        # Read the IR as a IENetwork
        try:
            self.network =  self.ie.read_network(model=model_xml, weights=model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        if args.cpu_extension and "CPU" in args.device:
            ie.add_extension(args.cpu_extension, "CPU")
            log.info("CPU extension loaded: {}".format(args.cpu_extension))

        if "CPU" in args.device:
            supported_layers = self.ie.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(args.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
                sys.exit(1)

        # Load the IENetwork into the plugin
        self.exec_network = self.ie.load_network(self.network, device, num_requests=num_requests)

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        log.info("Preparing input blobs")
        input_shape = []
        print("inputs number: " + str(len(self.network.input_info.keys())))

        for input_key in self.network.input_info:
            print("input shape: " + str(self.network.input_info[input_key].input_data.shape))
            print("input key: " + input_key)
            self.input_blob = input_key
            if len(self.network.input_info[input_key].input_data.layout) == 4:
                input_shape = self.network.input_info[input_key].input_data.shape

        return input_shape

    def get_output_name(self):
        '''
        Gets the input shape of the network
        '''
        log.info('Preparing output blobs')
        output_name, _ = "", self.network.outputs[next(iter(self.network.outputs.keys()))]
        for output_key in self.network.outputs:
            if self.network.layers[output_key].type == "DetectionOutput":
                self.output_blob, _ = output_key, self.network.outputs[output_key]
        
        if self.output_blob == "":
            log.error("Can't find a DetectionOutput layer in the topology")
            exit(-1)
        return self.output_blob

    def exec_net(self, image, request_id):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return

    
    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[request_id].wait(-1)
        return status


    def get_output(self, request_id):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[request_id].outputs[self.output_blob]