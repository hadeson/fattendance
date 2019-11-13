"""
Openvino age gender detection model
R50 model from insightface, train on asian faces
"""
import cv2
import os
import numpy as np
import configparser

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin


class AgeGenderEstimate(object):
    def __init__(self, plugin, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.plugin = plugin
        self.fd_net, self.fd_input_blob, _ = self.initOpenvinoModel()
        self.undefined_zone = float(self.config["AGE_GENDER_ASIA"]['Undefined_zone'])
        self.gender_label = ['Nu', 'Nam', '???']

    def initOpenvinoModel(self):
        pretrained_models_dir = self.config["AGE_GENDER_ASIA"]['Model_dir']
        pretrained_models_name = self.config["AGE_GENDER_ASIA"]['Model_name']
        model_path = os.path.join(pretrained_models_dir, pretrained_models_name)
        model_xml = model_path + ".xml"
        model_bin = model_path + ".bin"
        # Read IR
        net = IENetwork(model=model_xml, weights=model_bin)
        input_blob = next(iter(net.inputs))
        output_blob = next(iter(net.outputs))
        # Load network to the plugin
        exec_net = self.plugin.load(network=net)
        del net
        return exec_net, input_blob, output_blob

    def inference(self, img):
        finput = cv2.resize(img, (112, 112))
        finput = finput.transpose(2, 0, 1)
        finput = np.expand_dims(finput, axis=0)
        out = self.fd_net.infer(inputs={self.fd_input_blob: finput})['fc1'][0]
        # print(out.shape)
        age = np.sum(np.argmax(out[2:202].reshape((100, 2)), axis=1))
        gender_id = np.argmax(out[:2])
        if np.abs(out[0] - out[1]) <= self.undefined_zone:
            gender_id = 2
        return age, self.gender_label[gender_id]
