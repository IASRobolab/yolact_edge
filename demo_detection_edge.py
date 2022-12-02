import math
import os
import pdb
import sys

from pathlib import Path
yolact_path ='/root/yolact_edge'
sys.path.append(yolact_path)

camera_path = '/root/camera_utils'
sys.path.append(camera_path)

ai_path = '/root/ai_utils'
sys.path.append(ai_path)


from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.detectors.YolactEdgeInference import YolactEdgeInference

import numpy as np
import cv2
import time
import argparse


if __name__ == '__main__':
    #yolact_weights_new = str(Path.home()) + "/Desktop/yolact_edge_plus/weights/yolact_edge_plus_resnet50_box_penv_plenv_AI4M_79_800.pth"#yolact_edge_resnet50_54_800000.pth"
    #yolact_weights_new = str(Path.home()) + "/Downloads/yolact_edge_54_800000.pth"#yolact_edge_54_800000.pth"
    yolact_weights_new = "/root/yolact_edge/weights/yolact_edge_resnet50_54_800000.pth"#yolact_plus_resnet50_54_800000.pth"
    #yolact_weights_new = str(Path.home()) + "/Documents/yolact_edge/weights/yolact_edge_resnet50_box_penv_plenv_AI4M_79_800.pth"#yolact_plus_resnet50_54_800000.pth"
    #yolact_plus_resnet50_box_penv_plenv_AI4M_79_960.pth"
    # yolact_edge_resnet50_box_penv_plenv_AI4M_79_800.pth
    #yolact_weights_new = str(Path.home()) + "/Documents/robotic_arms_vision/weights/yolact_plus_resnet50_valve_39_520.pth"
    #yolact_edge_resnet50_54_800000.pth
    yolact_new = YolactEdgeInference(disable_tensorrt=True, model_weights=yolact_weights_new, score_threshold=0.5, display_img=True)


    camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD)


    while True:
        start_time = time.time()
        
        img = camera.get_rgb()
        img = np.array(img)
        

        #cv2.namedWindow("immagine", cv2.WINDOW_NORMAL)
        #cv2.imshow("immagine", img)

        yolact_infer = yolact_new.img_inference(img)
        #pdb.set_trace()
        
        
        print(1/(time.time()-start_time))
        

    




