"""
Openvino age gender detection model
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
        self.undefined_zone = float(self.config["AGE_GENDER"]['Undefined_zone'])
        self.gender_label = ['Nu', 'Nam', '???']

    def initOpenvinoModel(self):
        pretrained_models_dir = self.config["AGE_GENDER"]['Model_dir']
        pretrained_models_name = self.config["AGE_GENDER"]['Model_name']
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
        finput = cv2.resize(img, (62, 62))
        # finput = finput.swapaxes(1, 2).swapaxes(0, 1)
        finput = finput.transpose(2, 0, 1)
        finput = np.expand_dims(finput, axis=0)
        out = self.fd_net.infer(inputs={self.fd_input_blob: finput})
        age = np.squeeze(out['age_conv3']) * 100
        gender_prob = np.squeeze(out['prob'])
        cls_id = np.argmax(gender_prob)
        if np.abs(gender_prob[0] - gender_prob[1]) <= self.undefined_zone:
            cls_id = 2
        return age, self.gender_label[cls_id]
