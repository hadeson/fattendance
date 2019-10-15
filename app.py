import configparser
import sys, os
import time
import numpy as np
import cv2
import ast
from bson import ObjectId
from pymongo import MongoClient
from datetime import datetime
from openvino.inference_engine import IENetwork, IEPlugin
from arcface import FaceEmbedding
from mbnfacedet import MobileFaceDetect
from head_pose_est import HeadPoseEst
from face_search import bruteforce
from utils import initModel, area, cosineDistance, good_head_angle

class FaceRecognition(object):
    def __init__(self, parent=None):
        config = configparser.ConfigParser()
        config.read("config.ini")
        device = "MYRIAD"

        self.config = config
        self.face_database, self.face_collection = self.loadFaceData()
        self.log_database, self.log_collection = self.loadLogData()
        plugin = IEPlugin(device, plugin_dirs=None)
        self.face_embed = FaceEmbedding(plugin)
        self.face_detect = MobileFaceDetect(plugin)
        self.head_pose = HeadPoseEst(plugin)
        self.ch = int(config["CAMERA"]['Height'])
        self.cw = int(config["CAMERA"]['Width'])
        self.face_min_size = float(config["FACE_CONSOLIDATION"]['Face_min_ratio']) * (self.ch*self.cw)
        self.face_margin = int(config["FACE_CONSOLIDATION"]['Face_margin'])
        self.fm_threshold = float(config["FACE_MATCH"]['Face_threshold'])
        self.face_counter = int(config["FACE_MATCH"]['Counter'])
        self.ct_threshold = float(config["FACE_MATCH"]['Counter_threshold'])
        self.debug = int(config["FACE_MATCH"]['Debug'])
        self.no_face_frame_limit = int(config["DELAY"]['No_face_frame'])
        self.recognition_delay = float(config["DELAY"]['Recognition_delay'])
        self.angle_min = ast.literal_eval(config["HEAD_POSE"]["Angle_min"])
        self.angle_max = ast.literal_eval(config["HEAD_POSE"]["Angle_max"])
        self.log_img_dir = config["IMAGE_DIR"]['Log']
        if not os.path.exists(self.log_img_dir):
            os.mkdir(self.log_img_dir)

    def serve(self):
        # self.face_detection_phase = False
        face_rec_delay = time.time()
        no_face_frame = 0
        cam = cv2.VideoCapture(0)
        label = "Hello"
        while(True):
            ret, frame = cam.read()
            if not ret:
                # dead cam
                cv2.destroyAllWindows()
                return 1
            # go from standby to face detection phase
            # TODO change press key trigger to client-server trigger (e.g. IR sensor)
            if no_face_frame > self.no_face_frame_limit:
                # no error
                cv2.destroyAllWindows()
                return 0
            # face detection phase
            face_bboxes = self.face_detect.inference(frame)
            if len(face_bboxes) > 0:
                no_face_frame = 0
                areas = [area(box) for box in face_bboxes]
                max_id = np.argmax(np.asarray(areas))
                mfb = face_bboxes[max_id]
                # print(area(mfb), self.face_min_size, frame.shape)
                # face consolidation phase, calculate face angle
                if area(mfb) > self.face_min_size:
                    x0 = max(0, mfb[0] - self.face_margin)
                    y0 = max(0, mfb[1] - self.face_margin)
                    x1 = min(self.ch, mfb[2] + self.face_margin)
                    y1 = min(self.cw, mfb[3] + self.face_margin)
                    main_head = frame[y0:y1, x0:x1, :]
                    yaw, pitch, roll = self.head_pose.inference(main_head)
                    if not good_head_angle(yaw, pitch, roll, self.angle_min, self.angle_max):
                        # TODO: indicate on screen
                        cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
                        label = "please look at the camera"
                        text_color = (0, 255, 0)
                        cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, text_color, lineType=cv2.LINE_AA)
                        cv2.imshow("gandalf", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return 2
                        # TODO: face liveness detection
                        continue
                else:
                    # face too small
                    label = "please move closer"
                    text_color = (0, 255, 0)
                    cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, text_color, lineType=cv2.LINE_AA)
                    cv2.imshow("gandalf", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return 2
                    # print(label, area(mfb), face_min_size, cam_width, cam_height, mfb)
                    continue

                # face recognition phase
                face_rec_delay_amount = time.time() - face_rec_delay
                if face_rec_delay_amount >= self.recognition_delay:
                    main_face = frame[mfb[1]:mfb[3], mfb[0]:mfb[2], :]
                    # TODO face alignment
                    face_feature = self.face_embed.inference(main_face)
                    best_match = bruteforce(face_feature, self.face_database, self.fm_threshold)
                    # TODO face record
                    if best_match is None:
                        new_log = {
                            'result': 'failed',
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.updateLog(new_log, main_face)
                        label = "{}".format("Verification failed")
                    else:
                        new_log = {
                            'result': 'success',
                            'face_id': best_match['_id'],
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.updateLog(new_log, main_face)
                        label = "Verification success, hello {}!".format(str(best_match['name']))
                    face_rec_delay = time.time()
                # visualize for debug
                cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
                text_color = (0, 255, 0)
                if label[-1] != '!':
                    text_color = (0, 0, 255)

                cv2.putText(frame, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, text_color, lineType=cv2.LINE_AA)
            else:
                no_face_frame += 1
                print(no_face_frame, "/", self.no_face_frame_limit)

            cv2.imshow("gandalf", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return 2    

            
    def loadFaceData(self):
        """
        return cursor and collection
        """
        url = self.config["MONGO"]['Url']
        port = int(self.config["MONGO"]['Port'])
        db_name = self.config["MONGO"]['Database']
        col_name = self.config["MONGO"]['FaceCollection']
        client = MongoClient(url, port)
        db = client[db_name]
        collection = db[col_name]
        # get the whole collection
        people = list(collection.find())
        return people, collection

    def loadLogData(self):
        """
        return collection
        """
        url = self.config["MONGO"]['Url']
        port = int(self.config["MONGO"]['Port'])
        db_name = self.config["MONGO"]['Database']
        col_name = self.config["MONGO"]['LogCollection']
        client = MongoClient(url, port)
        db = client[db_name]
        collection = db[col_name]
        # get the whole collection
        logs = list(collection.find())
        return logs, collection

    def updateLog(self, new_log, face_img):
        p_id = self.log_collection.insert_one(new_log).inserted_id
        self.log_collection.update_one({'_id': p_id}, {"$set": new_log}, upsert=False)
        new_img_name = "{}_{}.jpg".format(new_log['time'], new_log['result'])
        # remove special chars
        new_img_name = new_img_name.replace("-", "_")
        new_img_name = new_img_name.replace(":", "_")
        new_img_name = new_img_name.replace(" ", "_")
        new_img_path = os.path.join(self.log_img_dir, new_img_name)
        # print(new_img_path)
        cv2.imwrite(new_img_path, face_img)
        return p_id