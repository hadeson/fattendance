import configparser
import sys, os
import time
import numpy as np
import cv2
import ast
import requests, json
import socket, select
from bson import ObjectId
from pymongo import MongoClient
from datetime import datetime
from openvino.inference_engine import IENetwork, IEPlugin
from arcface import FaceEmbedding
from mbnfacedet import MobileFaceDetect
from head_pose_est import HeadPoseEst
from face_search import bruteforce
from utils import initModel, area, cosineDistance, good_head_angle
from draw import DisplayDraw

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
        self.door_host = config["DOOR_CONTROLLER"]['Host']
        self.door_port = int(config["DOOR_CONTROLLER"]['Port'])
        self.door_name = config["DOOR_CONTROLLER"]['Name']
        self.door_url = "http://{}:{}{}".format(self.door_host, self.door_name, self.door_port)
        self.door_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.door_socket.connect((self.door_host, self.door_port))
        self.open_door_signal = config["DOOR_CONTROLLER"]['Open_door_signal']
        self.open_door_signal = self.open_door_signal.encode()
        
        self.face_lap_min_score = float(config["MOBILE_FACE_DET"]['Laplacian_min_score'])
        self.face_min_size = float(config["FACE_CONSOLIDATION"]['Face_min_ratio']) * (self.ch*self.cw)
        self.face_margin = int(config["FACE_CONSOLIDATION"]['Face_margin'])
        self.fm_threshold = float(config["FACE_MATCH"]['Face_threshold'])
        # self.face_counter = int(config["FACE_MATCH"]['Counter'])
        # self.ct_threshold = float(config["FACE_MATCH"]['Counter_threshold'])
        self.debug = int(config["FACE_MATCH"]['Debug'])
        self.no_face_frame_limit = int(config["DELAY"]['No_face_frame'])
        self.recognition_delay = float(config["DELAY"]['Recognition_delay'])
        self.recog_suc_delay = float(config["DELAY"]['Recognize_success'])
        self.recog_fai_delay = float(config["DELAY"]['Recognize_failed'])
        self.angle_min = ast.literal_eval(config["HEAD_POSE"]["Angle_min"])
        self.angle_max = ast.literal_eval(config["HEAD_POSE"]["Angle_max"])
        self.log_img_dir = config["IMAGE_DIR"]['Log']
        self.window_mult = float(config["DISPLAY"]['Window_mult'])
        self.max_signal_send = int(config["DOOR_CONTROLLER"]['Max_signal_send'])
        self.display_draw = DisplayDraw()
        self.test_face = False
        self.face_img = cv2.imread("test_img/test_face.jpg")
        self.face_img = cv2.resize(self.face_img, (120, 180))
        if not os.path.exists(self.log_img_dir):
            os.mkdir(self.log_img_dir)

    def serve(self):
        face_rec_delay = time.time()
        no_face_frame = 0
        cv2.namedWindow('gandalf', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('gandalf', int(self.cw * self.window_mult), int(self.ch * self.window_mult))
        cv2.setWindowProperty('gandalf', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cam = cv2.VideoCapture(0)
        # cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        # label = "Hello"
        while(True):
            ret, frame = cam.read()
            # h, w, c = frame.shape
            # print(h, w, c)
            if not ret:
                # dead cam
                cam = cv2.VideoCapture(0)
                time.sleep(3.000) # some delay to init cam
                # cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
                continue
                # cv2.destroyAllWindows()
                # return 1
            if self.test_face:
                frame[120:300, 280:400, :] = self.face_img
                # add test face on frame

            # go from standby to face detection phase
            # if no_face_frame > self.no_face_frame_limit:
            #     # no error
            #     cv2.destroyAllWindows()
            #     return 0
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

                    # detect blurry face
                    h, w, c = main_head.shape
                    # print("hp img shape: ", img.shape)
                    if (h > 0) and (w > 0):
                        blur_face = cv2.resize(main_head, (112, 112))
                        blur_face_var = cv2.Laplacian(blur_face, cv2.CV_64F).var()
                        if blur_face_var < self.face_lap_min_score:
                            cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
                            face_rec_delay_amount = time.time() - face_rec_delay
                            if face_rec_delay_amount > self.recognition_delay:
                                frame = self.display_draw.drawLACText(frame)
                            else:
                                frame = self.display_draw.drawLastText(frame)
                                # label = "please look at the camera"
                            cv2.imshow("gandalf", frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                return 2
                            continue

                    # detect head pose
                    yaw, pitch, roll = self.head_pose.inference(main_head)
                    if not good_head_angle(yaw, pitch, roll, self.angle_min, self.angle_max):
                        cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
                        face_rec_delay_amount = time.time() - face_rec_delay
                        if face_rec_delay_amount > self.recognition_delay:
                            frame = self.display_draw.drawLACText(frame)
                        else:
                            frame = self.display_draw.drawLastText(frame)
                        cv2.imshow("gandalf", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return 2
                        continue

                    # TODO: face liveness detection
                else:
                    # face too small
                    face_rec_delay_amount = time.time() - face_rec_delay
                    if face_rec_delay_amount > self.recognition_delay:
                        frame = self.display_draw.drawMCText(frame)
                    else:
                        frame = self.display_draw.drawLastText(frame)
                    cv2.imshow("gandalf", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return 2
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
                        self.display_draw.drawFailedText(frame)
                        self.recognition_delay = self.recog_fai_delay
                    else:
                        self.callDoorControllerSocket()
                        new_log = {
                            'result': 'success',
                            'face_id': best_match['_id'],
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.updateLog(new_log, main_face)
                        self.display_draw.drawSuccessText(frame, str(best_match['name']))
                        self.recognition_delay = self.recog_suc_delay
                    face_rec_delay = time.time()
                cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
            else:
                no_face_frame += 1
                face_rec_delay_amount = time.time() - face_rec_delay
                if face_rec_delay_amount > self.recognition_delay:
                    frame = self.display_draw.drawDefaultText(frame)
                else:
                    frame = self.display_draw.drawLastText(frame)
            cv2.imshow("gandalf", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return 2    
            elif cv2.waitKey(1) & 0xFF == ord('t'):
                self.test_face = True
            elif cv2.waitKey(1) & 0xFF == ord('y'):
                self.test_face = False

            
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

    def callDoorControllerPost(self, signal):
        body = {
            'signal': signal,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        header = {"Content-type": "application/x-www-form-urlencoded",
                "Accept": "text/plain"} 
        body_json = json.dumps(body)
        r = requests.post(self.door_url, data=body_json, headers=header)
        print(r.json())

    def callDoorControllerSocket(self):
        # check connection
        try:
            ready_to_read, ready_to_write, in_error = \
                select.select([self.door_socket,], [self.door_socket,], [], 5)
        except select.error:
            print(select.error)
            # shutdown connection
            self.door_socket.shutdown(2)    # 0 = done receiving, 1 = done sending, 2 = both
            self.door_socket.close()
            self.door_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # try to reconnect
            self.door_socket.connect((self.door_host, self.door_port))
        for i in range(self.max_signal_send):
            print("sending to door controller")
            self.door_socket.sendall(self.open_door_signal)
                # data = s.recv(1024)
                # data_str = data.decode('utf-8')
                # if data_str == 'ok':
                #     break


if __name__ == '__main__':
    face_recog = FaceRecognition()
    face_recog.serve()
