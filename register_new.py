"""
face registration
"""
import cv2
import time
import os
import pickle
import numpy as np
import configparser
import pymongo
from mbnfacedet import MobileFaceDetect
from arcface import FaceEmbedding
from face_search import bruteforce
from utils import initModel, area, loadMongoData
from openvino.inference_engine import IENetwork, IEPlugin

def main():
    cam = cv2.VideoCapture(0)
    # load data
    config = configparser.ConfigParser()
    config.read("config.ini")
    face_database, face_collection = loadMongoData(config)
    # load models
    device = "MYRIAD"
    data_dir = "data"
    plugin = IEPlugin(device, plugin_dirs=None)
    face_embed = FaceEmbedding(plugin)
    face_detect = MobileFaceDetect(plugin)
    # params
    fd_conf = 0.5
    fm_threshold = 0.6
    label = "new"
    period = 1
    button_pressed = False
    max_num = 3
    num = 0
    face_features = []
    face_imgs = []

    s = time.time()
    while(True):
        ret, frame = cam.read()
        if not ret:
            print("dead cam")
            cam = cv2.VideoCapture(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        face_bboxes = []
        # if time.time() - s > period:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            button_pressed = True
        if button_pressed and (num < max_num):
            face_bboxes = face_detect.inference(frame)
        if (len(face_bboxes) > 0) and button_pressed:
            areas = [area(box) for box in face_bboxes]
            max_id = np.argmax(np.asarray(areas))
            mfb = face_bboxes[max_id]
            main_face = frame[mfb[1]:mfb[3], mfb[0]:mfb[2], :]
            # TODO real face detection
            # TODO face alignment
            face_feature = face_embed.inference(main_face)
            face_feature = face_feature.tolist()
            face_features.append(face_feature)
            face_imgs.append(main_face)
            num += 1
            button_pressed = False
            s = time.time()
            # visualize for debug
            cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
            cv2.putText(frame, str(num), (mfb[0], mfb[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), lineType=cv2.LINE_AA)
            print(num)

        if num >= max_num:
            # add new face features to database
            new_id = face_database.count()
            new_face = {
                'name': str(new_id),
                'feats': face_features
            }
            p_id = face_collection.insert_one(new_face).inserted_id
            # commit changes
            face_collection.update_one({'_id': p_id}, {"$set": new_face}, upsert=False)

            # save images
            img_dir = os.path.join(data_dir, str(new_id))
            os.mkdir(img_dir)
            for i, face in enumerate(face_imgs):
                img_path = os.path.join(img_dir, "{}.jpg".format(i))
                cv2.imwrite(img_path, face)
            face_imgs = []
            face_features = []
            num = 0
            s = time.time()
            print("done!")

        cv2.imshow("face registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

if __name__ == '__main__':
    main()