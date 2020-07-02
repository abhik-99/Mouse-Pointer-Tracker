import os
import logging
import numpy as np
import cv2

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import input_feeder

def build_argparser():
    '''
    Parser Required Arguments:-
    1. Path to Face Detection Model.
    2. Path to Facial Landmark Model.
    3. Path to Head Pose Detection Model.
    4. Path to Gaze Estimation Model.
    5. Path to the Input (Video/Image). (Here only video is considered)

    Parser Optional Arguments:-
    1. Device (default CPU).
    2. CPU Extension (default None).
    3. Probability Threshold (default 0.6).

    '''
    parser = ArgumentParser(description = 'This is the main file to control the mouse pointer from the video input. Please execute it with the arguments.')
    
    #Required Arguments
    parser.add_argument('-i', '--input', required = True, type = str,
                        help = 'Path to video file or enter cam for webcam')
    parser.add_argument('-fd', '--face_detection_model', required = True, type = str,
                        help = 'Path to .xml file of Face Detection model.')
    parser.add_argument('-fl', '--facial_landmark_model', required = True, type = str,
                        help = 'Path to .xml file of Facial Landmark Detection model.')
    parser.add_argument('-hp', '--head_pose_model', required = True, type = str,
                        help = 'Path to .xml file of Head Pose Estimation model.')
    parser.add_argument('-ge', '--gaze_estimation_model', required = True, type = str,
                        help = 'Path to .xml file of Gaze Estimation model.')
    
    #Optional Arguments
    parser.add_argument('-flags', '--preview_flags', required = False, nargs = '+',
                        default = [],
                        help = 'Accepted values - \'fd\' for Face Detection, \'fld\' for Facial Landmark Detection,' 
                             '\'hp\' for Head Pose Estimation, \'ge\' for Gaze Estimation.'
                             'This option will help you see the individual output from the model. eg:- -flags fld hp' )
    parser.add_argument('-l', '--cpu_extension', required = False, type = str,
                        default = None,
                        help = 'Path to the CPU Extension')
    parser.add_argument('-pt', '--prob_threshold', required = False, type = float,
                        default = 0.6,
                        help = 'Probability threshold for model to detect the face accurately from the video frame.')
    parser.add_argument('-d', '--device', type = str, default = 'CPU',
                        help = 'The target device to infer on: '
                             'CPU, GPU, FPGA or MYRIAD is acceptable. Sample '
                             'will look for a suitable plugin for device '
                             'specified (CPU by default). Please note that only CPU is available in Author\'s Workstation')
    parser.add_argument('-o', '--output_format', type = str, default = 'file', required = False, 
                        help = 'If the parameter is \'file\', a file of name \'output.mp4\' is created in the current directory.'
                                'If the parameter is \'visual\', the frames are displayed.')
    
    return parser



def main():

    # Grab command line args
    args = build_argparser().parse_args()
    preview_flags = args.preview_flags
    
    logger = logging.getLogger()
    input_path = args.input

    if input_path.lower() == 'cam':
        input_feed = input_feeder('cam')
    else:
        if not os.path.isfile(input_path):
            logger.error('Unable to find specified video file')
            exit(1)
        input_feed = input_feeder('video', input_path)
    
    modelPathDict = {'face_detect':args.facedetectionmodel, 'face_landmark_regress':args.faciallandmarkmodel, 
                    'head_pose':args.headposemodel, 'gaze_estimate':args.gazeestimationmodel}
    
    for pathname, filepath in modelPathDict:
        if not os.path.isfile(modelPathDict[filepath]):
            logger.error('Unable to find specified '+pathname+' xml file')
            exit(1)

    #initializing models  
    fdm = FaceDetectionModel(modelPathDict['face_detect'], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(modelPathDict['face_landmark_regress'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(modelPathDict['head_pose'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict['gaze_estimate'], args.device, args.cpu_extension)
    
    #initializing mouse controller
    mouse_controller = MouseController('medium','fast')
    
    input_feed.load_data()

    #checking models 
    fdm.check_model()
    fldm.check_model()
    hpem.check_model()
    gem.check_model()

    #loading models / creating executable network
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    
    frame_count = 0
    for ret, frame in input_feed.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        """
        Sequence of model execution:-
        1. Predict from each model.
        2. Preprocess of outputs from each model.
        3. Send the processed output to the next model.

        Model Sequence:- 
                                - Head Pose Estimation Model      -
        Face Detection Model <                                      > Gaze Estimation Model 
                                - Facial Landmark Detection Model -  
        """

        croppedFace, _ = fdm.predict(frame.copy(), args.prob_threshold)

        if type(croppedFace)==int:
            logger.error('Unable to detect the face.')
            if key==27:
                break
            continue
        
        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
        
        if (not len(preview_flags)==0):
            preview_frame = frame.copy()
            if 'fd' in preview_flags:
                #cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
                preview_frame = croppedFace
            if 'fld' in preview_flags:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                
            if 'hp' in preview_flags:
                cv2.putText(preview_frame, 'Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}'.format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            
            if 'ge' in preview_flags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le = cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                
        cv2.imshow('Sample',cv2.resize(preview_frame,(500,500)))
        
        #move the mouse pointer 
        mouse_controller.move(new_mouse_coord[0],new_mouse_coord[1])  

        if key==27:
                break

    logger.error('VideoStream ended...')
    cv2.destroyAllWindows()
    input_feeder.close()
     
    

if __name__ == '__main__':
    main() 