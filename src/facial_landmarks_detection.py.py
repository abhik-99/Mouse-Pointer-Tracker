'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name, self.device = model_name, device
        self.core = IECore()
        
        if device == 'CPU' and extensions != None:
            self.core.add_extension(extension_path = extensions, device = device)

        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.model = self.core.read_network(model = model_xml, weights = model_bin)

        return

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        
        self.input_name = next(iter(self.model.inputs))
        self.output_name = next(iter(self.model.outputs))

        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_shape = self.model.outputs[self.output_name].shape

        self.net = self.core.load_network(self.model, self.device)

        return self.net

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img = self.preprocess_input(image)

        return self.net.infer({self.input_name: img})

    def check_model(self):
        supported_layers = self.core.query_network(network = self.model, device = self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")

        return

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
