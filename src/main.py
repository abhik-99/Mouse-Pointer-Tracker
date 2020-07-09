import os
import sys
import logging
import numpy as np
import cv2
import pyautogui

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

FRAME_WIDTH = 600
FRAME_HEIGHT = 600

GAZE_ARROW_WIDTH = 5
GAZE_ARROW_LENGTH = 100


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
    parser.add_argument('-o', '--output_file', type = str, default = 'y', required = False, 
                        help = 'If specified, then the output file by the name \'output.<extension>\''
                        'is generated in the ./src directory. This file shows the detection output.'
                        'Accepted Values:- [y(default), n]. '
                        'Please note that this option is only available for Linux at the moment.')
    parser.add_argument('-z', '--zoomed', type = bool, default = False, required= False, 
                        help = 'If True then displays only the cropped face in the video. The output file, however, shows full image.')
    
    return parser



def main():
    args = build_argparser().parse_args()

    preview_flags = args.preview_flags
    
    logger = logging.getLogger()
    input_path = args.input

    if input_path.lower() == 'cam':
        input_feed = InputFeeder('cam')
    else:
        if not os.path.isfile(input_path):
            logger.error('Unable to find specified video file')
            exit(1)
        file_extension = input_path.split(".")[-1]
        if(file_extension in ['jpg', 'jpeg', 'bmp']):
            input_feed = InputFeeder('image', input_path)
        elif(file_extension in ['avi', 'mp4']):
            input_feed = InputFeeder('video', input_path)
        else:
            logger.error("Unsupported file Extension. Allowed ['jpg', 'jpeg', 'bmp', 'avi', 'mp4']")
            exit(1)

    if sys.platform == "linux" or sys.platform == "linux2":
        #CODEC = 0x00000021
        CODEC = cv2.VideoWriter_fourcc(*"mp4v")
    elif sys.platform == "darwin":
        CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
    else:
        print("Unsupported OS.")
        exit(1)

    file_flag = False
    if args.output_file.lower() == 'y':
        file_flag = True
        out = cv2.VideoWriter('output.mp4', CODEC, 30, (FRAME_WIDTH, FRAME_HEIGHT))
    
    modelPathDict = {'face_detect':args.face_detection_model, 'face_landmark_regress':args.facial_landmark_model, 
                    'head_pose':args.head_pose_model, 'gaze_estimate':args.gaze_estimation_model}
    
    for pathname in modelPathDict:
        if not os.path.isfile(modelPathDict[pathname]):
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
    
        key = cv2.waitKey(60)

        """
        Sequence of model execution:-
        1. Predict from each model.
        2. Preprocess of outputs from each model.
        3. Send the processed output to the next model.

        Model Sequence:- 
                                -   Head Pose Estimation Model      -
        Face Detection Model <(First Head Pose and Then Facial Landmark)>Gaze Estimation Model 
                                -   Facial Landmark Detection Model -  
        """

        cropped_face, face_coords = fdm.preprocess_output(frame.copy(), fdm.predict(frame.copy()), args.prob_threshold)

        if type(cropped_face)==int:
            logger.error('Unable to detect the face.')
            if key==27:
                break
            continue
        
        hp_out = hpem.preprocess_output(hpem.predict(cropped_face.copy()))
        
        left_eye, right_eye, eye_coords = fldm.preprocess_output(cropped_face.copy(), fldm.predict(cropped_face.copy()))
        
        new_mouse_coord, gaze_vector = gem.preprocess_output(gem.predict(left_eye, right_eye, hp_out), hp_out)
        
        if (not len(preview_flags) == 0) or file_flag:
            preview_frame = frame.copy()
            

            if 'fd' in preview_flags:
                preview_frame = cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (0,0,255), 3)
                cropped_face = preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
            
            if 'fld' in preview_flags:
                cropped_face = cv2.rectangle(cropped_face, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cropped_face = cv2.rectangle(cropped_face, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                
                preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_face
                
            if 'hp' in preview_flags:
                cv2.putText(preview_frame, 'Pose Angles: yaw: {:.2f} | pitch: {:.2f} | roll: {:.2f}'.format(hp_out[0],hp_out[1],hp_out[2]), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            if 'ge' in preview_flags:

                x, y = int(gaze_vector[0] *GAZE_ARROW_LENGTH) , - int(gaze_vector[1]*GAZE_ARROW_LENGTH)
                
                
                le_mid_x = int((eye_coords[0][0] + eye_coords[0][2])/2)
                le_mid_y = int((eye_coords[0][1] + eye_coords[0][3])/2)
                re_mid_x = int((eye_coords[1][0] + eye_coords[1][2])/2)
                re_mid_y = int((eye_coords[1][1] + eye_coords[1][3])/2)

                cv2.arrowedLine(cropped_face, (le_mid_x, le_mid_y),
                ((le_mid_x + x), (le_mid_y + y)),
                (255, 0 , 0), GAZE_ARROW_WIDTH)
                cv2.arrowedLine(cropped_face, (re_mid_x, re_mid_y),
                ((re_mid_x + x), (re_mid_y + y)),
                (255, 0, 0), GAZE_ARROW_WIDTH)

                preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_face
                
           
            if(not len(preview_flags) == 0) and frame_count %2 == 0:
                if args.zoomed:
                    cv2.imshow('Cropped Face',cv2.resize(cropped_face,(FRAME_WIDTH, FRAME_HEIGHT)))
                else:
                    cv2.imshow('Preview',cv2.resize(preview_frame,(FRAME_WIDTH, FRAME_HEIGHT)))
                
            if file_flag:
                out.write(cv2.resize(preview_frame, (FRAME_WIDTH, FRAME_HEIGHT)))
        
        #move the mouse pointer 
        try:
            mouse_controller.move(new_mouse_coord[0],new_mouse_coord[1])
        except pyautogui.FailSafeException:
            pass
        
        if frame_count%2==0 and len(preview_flags) == 0:
            cv2.imshow('Video',cv2.resize(frame,(FRAME_WIDTH, FRAME_HEIGHT)))

        if key==27:
                break

    logger.error('VideoStream ended.')
    if args.output_file.lower() == 'y': 
        out.release()
    input_feed.close()
    cv2.destroyAllWindows()    
     
    

if __name__ == '__main__':
    main() 