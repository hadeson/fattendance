import cv2
import numpy as np
import os
import shutil
import configparser
from bson import ObjectId
from pymongo import MongoClient
from openvino.inference_engine import IENetwork, IEPlugin
from arcface import FaceEmbedding
from mbnfacedet import MobileFaceDetect
from face_search import bruteforce

def loadFaceData():
    """
    return cursor and collection
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    url = config["MONGO"]['Url']
    port = int(config["MONGO"]['Port'])
    db_name = config["MONGO"]['Database']
    col_name = config["MONGO"]['FaceCollection']
    client = MongoClient(url, port)
    db = client[db_name]
    collection = db[col_name]
    # get the whole collection
    # people = list(collection.find())
    return collection

def main():
    src_dir = "new_register"
    dst_dir = "data"
    face_collection = loadFaceData()
    device = "MYRIAD"
    plugin = IEPlugin(device, plugin_dirs=None)
    face_embed = FaceEmbedding(plugin)
    cos_dst = 0.5
    for subdir, dirs, files in os.walk(src_dir):
        for d in dirs:
            print(d)
            dir_path = os.path.join(src_dir, d)
            feats = []
            name = d
            for sd, ds, files in os.walk(dir_path):
                for f in files:
                    file_path = os.path.join(sd, f)
                    img = cv2.imread(file_path)
                    fe = face_embed.inference(img)
                    feats.append(fe.tolist())
            new_face = {
                'name': name,
                'feats': feats
            }
            # insert to Mongo
            p_id = face_collection.insert_one(new_face).inserted_id
            # commit changes
            face_collection.update_one({'_id': p_id}, {"$set": new_face}, upsert=False)
            # create account imgs dir
            new_id = str(p_id)
            img_dir = os.path.join(dst_dir, str(new_id))
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            # paste images to account dir
            for sd, ds, files in os.walk(dir_path):
                for f in files:
                    src_file = os.path.join(sd, f)
                    dst_file = os.path.join(img_dir, f)
                    print(dst_file)
                    shutil.copyfile(src_file, dst_file)
            print("done!")
            



if __name__ == '__main__':
    main()