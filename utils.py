import cv2
import os
import pickle
import time, ast
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin

def initModel(model_xml, model_bin, plugin):
    # cpu_ext = "/opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libcpu_extension.so"
    # device = "CPU"
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net)
    del net
    return exec_net, input_blob, output_blob

def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def cosineDistance(a, b):
    '''return cosine distance of 2 vectors'''
    a = a.flatten()
    b = b.flatten()
    ab = np.matmul(np.transpose(a), b)
    aa = np.sqrt(np.sum(np.multiply(a, a)))
    bb = np.sqrt(np.sum(np.multiply(b, b)))
    ret = 1 - (ab / (aa * bb))
    return ret

def loadData(embed_file_path):
    face_embed_database = []
    if os.path.isfile(embed_file_path):
        face_embed_database = pickle.load(open(embed_file_path, "rb"))
    return face_embed_database

def good_head_angle(y, p, r, angle_min, angle_max):
    """
    good head angle would be looking directly to the camera, give or take 
    some degree each
    TODO: implement it
    """
    if ((angle_min[0] < y) and (angle_max[0] > y) 
        and (angle_min[1] < p) and (angle_max[1] > p) 
        and(angle_min[2] < r) and (angle_max[2] > r)):
        return True
    return False
    
def loadFaceData(config):
    """
    return Cursor and full collection
    """
    url = config["MONGO"]['Url']
    port = int(config["MONGO"]['Port'])
    db_name = config["MONGO"]['Database']
    col_name = config["MONGO"]['FaceCollection']
    client = MongoClient(url, port)
    db = client[db_name]
    collection = db[col_name]
    # get the whole collection
    people = list(collection.find())
    return people, collection

def loadLogData(config):
    """
    return Cursor and full collection
    """
    url = config["MONGO"]['Url']
    port = int(config["MONGO"]['Port'])
    db_name = config["MONGO"]['Database']
    col_name = config["MONGO"]['LogCollection']
    client = MongoClient(url, port)
    db = client[db_name]
    collection = db[col_name]
    # get the whole collection
    # people = list(collection.find())
    return collection