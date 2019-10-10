"""
A quick hack together serve forever:
    + face detection 
    + embedding 
    + matching search
    + send request to door controller
"""
import cv2
import numpy as np
import time
import configparser
from arcface import FaceEmbedding
from mbnfacedet import MobileFaceDetect
from face_search import bruteforce
from utils import initModel, area, loadData
from openvino.inference_engine import IENetwork, IEPlugin

def main():
    cam = cv2.VideoCapture(0)
    # load data
    embed_file_path = "data/embed.pkl"
    face_database = loadData(embed_file_path)
    # load models
    device = "MYRIAD"
    plugin = IEPlugin(device, plugin_dirs=None)

    face_embed = FaceEmbedding(plugin)
    face_detect = MobileFaceDetect(plugin)
    # params
    config = configparser.ConfigParser()
    config.read("config.ini")
    fm_threshold = float(config["FACE_MATCH"]['Threshold'])
    label = "new"

    while(True):
        ret, frame = cam.read()
        if not ret:
            print("dead cam")
            cam = cv2.VideoCapture(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        face_bboxes = face_detect.inference(frame)
        if len(face_bboxes) > 0:
            areas = [area(box) for box in face_bboxes]
            max_id = np.argmax(np.asarray(areas))
            mfb = face_bboxes[max_id]
            main_face = frame[mfb[1]:mfb[3], mfb[0]:mfb[2], :]
            # TODO real face detection
            # TODO face alignment
            # face_feature = face_embed(main_face, fe_net, fe_input_blob)
            # s = time.time()
            face_feature = face_embed.inference(main_face)
            # print(time.time() - s)
            # TODO face record
            best_match = bruteforce(face_feature, face_database, fm_threshold)
            if best_match is None:
                label = "new"
            else:
                label = str(best_match['id'])
            # visualize for debug
            cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
            cv2.putText(frame, label, (mfb[0], mfb[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.imshow("gandalf", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

if __name__ == '__main__':
    main()