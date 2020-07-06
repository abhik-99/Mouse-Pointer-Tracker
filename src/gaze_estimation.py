
import os
import numpy as np
import cv2
from openvino.inference_engine import IECore

class GazeEstimationModel:
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
        self.input_names = [i for i in self.model.inputs.keys()]
        self.output_names = [i for i in self.model.outputs.keys()]


        self.input_shape = self.model.inputs[self.input_names[1]].shape

        self.net = self.core.load_network(self.model, self.device)

        return self.net

    def predict(self, head_pose_angles, l_eye, r_eye):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        l_eye_processed, r_eye_processed = self.preprocess_input(l_eye.copy()), self.preprocess_input(r_eye.copy())

        return self.net.infer({'head_pose_angles':head_pose_angles, 'left_eye_image':l_eye_processed, 'right_eye_image':r_eye_processed})

    def check_model(self):
        supported_layers = self.core.query_network(network = self.model, device_name = self.device)
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

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        #gaze_vector = gaze_vector / cv2.norm(gaze_vector)
        rollValue = hpa[2] #angle_r_fc output from HeadPoseEstimation model
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector
