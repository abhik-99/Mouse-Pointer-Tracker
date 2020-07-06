
import os
import numpy as np
import cv2
from openvino.inference_engine import IECore

class FacialLandmarksDetectionModel:
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
        img = self.preprocess_input(image.copy())

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

    def preprocess_output(self, image, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = outputs[self.output_names][0]
 
        coords = (outs[0].tolist()[0][0], outs[1].tolist()[0][0], outs[2].tolist()[0][0], outs[3].tolist()[0][0])
        
        h=image.shape[0]
        w=image.shape[1]

        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) #(lefteye_x, lefteye_y, righteye_x, righteye_y)
        
        re_xmin=coords[2]-10
        re_ymin=coords[3]-10
        re_xmax=coords[2]+10
        re_ymax=coords[3]+10
        
        le_xmin=coords[0]-10
        le_ymin=coords[1]-10
        le_xmax=coords[0]+10
        le_ymax=coords[1]+10

        #cv2.rectangle(image,(le_xmin,le_ymin),(le_xmax,le_ymax),(255,0,0))
        #cv2.rectangle(image,(re_xmin,re_ymin),(re_xmax,re_ymax),(255,0,0))
        #cv2.imshow("Image",image)

        left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
        
        return left_eye, right_eye, eye_coords
