'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
import cv2
from openvino.inference import IECore

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
        img = cv2.resize(image, tuple(self.input_shape[2:][::-1])) #resizing the image
        img = img.transpose((2,0,1)) #brining channel to the front.
        img = img.reshape(1, *img.shape)

        return img

    def preprocess_output(self, image, outputs, prob_threshold = 0.6):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        outs = outputs[self.output_name][0][0]
        for each in outs:
            conf = each[2]
            if conf>prob_threshold:
                x_min = each[3]
                y_min = each[4]
                x_max = each[5]
                y_max = each[6]
                coords.append([x_min,y_min,x_max,y_max])

        if (len(coords) == 0):
            return 0, 0

        coords = coords[0] #Selecting only the first face detected

        h = image.shape[0]
        w = image.shape[1]

        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face, coords
