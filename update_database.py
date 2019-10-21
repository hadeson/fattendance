"""
update existing database using newly captured face
swap new face with oldest one when reaching maximum number of faces per account
"""
import cv2
import numpy as np
import configparser
import os
from datetime import datetime, timedelta
from bson import ObjectId
from pymongo import MongoClient
from arcface import FaceEmbedding
from openvino.inference_engine import IENetwork, IEPlugin
from utils import cosineDistance

class FaceUpdate(object):
    def __init__(self, parent=None):
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config
        self.face_db, self.face_col = self.loadFaceData()
        self.log_db, self.log_col = self.loadLogData()
        self.face_img_dir = config["IMAGE_DIR"]['Employee']
        self.log_img_dir = config["IMAGE_DIR"]['Log']
        self.face_lap_min_score = float(config["MOBILE_FACE_DET"]['Laplacian_min_score_update'])
        self.fm_threshold = float(config["FACE_MATCH"]['Face_update_threshold_max'])
        self.fm_identical_threshold = float(config["FACE_MATCH"]['Face_update_threshold_min'])
        self.max_img_per_acc = int(config["MONGO"]['Max_img_per_acc'])
        device = "MYRIAD"
        plugin = IEPlugin(device, plugin_dirs=None)
        self.face_embed = FaceEmbedding(plugin)

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

    def dailyUpdate(self, prev_days=1):
        """
        update all faces in defined previous days
        """
        epoch = datetime.now() - timedelta(days=prev_days)
        epoch_str = epoch.strftime('%Y-%m-%d %H:%M:%S')
        # success_logs = [log for log in self.log_db if log['result'] == 'success']
        success_logs = [
            log for log in self.log_db 
            if (log['result'] == 'success') and (log['time'] > epoch_str)
        ]
        for log in success_logs:
            # load face based on log
            log_id = str(log['face_id'])
            face_acc = self.face_col.find_one({"_id": ObjectId(log_id)})
            # get associated log image
            log_img_name = "{}_{}.jpg".format(log['time'], log['result'])
            # remove special chars
            log_img_name = log_img_name.replace("-", "_")
            log_img_name = log_img_name.replace(":", "_")
            log_img_name = log_img_name.replace(" ", "_")
            log_img_path = os.path.join(self.log_img_dir, log_img_name)
            log_img = cv2.imread(log_img_path)
            # calculate blurness
            blur_face = cv2.resize(log_img, (112, 112))
            blur_face_var = cv2.Laplacian(blur_face, cv2.CV_64F).var()
            if blur_face_var < self.face_lap_min_score:
                print(log_img_path, " : face too blurry")
                continue
            # calculate arcface score
            log_face_feat = self.face_embed.inference(log_img)
            # get smallest cosine distance
            min_dst = 100
            best_feat = None
            for feat in face_acc['feats']:
                feat_np = np.asarray(feat)
                cos_dst = cosineDistance(log_face_feat, feat_np)
                if (cos_dst < self.fm_threshold) and (cos_dst < min_dst):
                    min_dst = cos_dst
                    best_feat = feat_np

            if min_dst > self.fm_threshold:
                print(log_img_path, " : face too different", min_dst)
                continue
            elif min_dst < self.fm_identical_threshold:
                print(log_img_path, " : face too similar", min_dst)
                continue
            
            print(log_img_path, " : added!")
            # update database, if number of imgs per acc exceed max allowance,
            # delete first recorded feat, keep image
            face_acc['feats'].append(best_feat.tolist())
            if len(face_acc['feats']) > self.max_img_per_acc:
                old_feat = face_acc['feats'].pop(0)
            self.face_col.update_one({'_id': face_acc['_id']}, {"$set": face_acc}, upsert=False)
            # save new image
            face_acc_img_dir = os.path.join(self.face_img_dir, str(face_acc['_id']))
            new_log_img_path = os.path.join(face_acc_img_dir, log_img_name)
            cv2.imwrite(new_log_img_path, log_img)


if __name__ == '__main__':
    updater = FaceUpdate()
    updater.dailyUpdate(prev_days=4)