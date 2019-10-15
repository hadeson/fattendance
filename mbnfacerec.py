"""
Openvino face recognition model
mobilenet backbone
"""
import cv2
import os
import numpy as np
import configparser

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin


class FaceEmbeddingLite(object):
    def __init__(self, plugin, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.plugin = plugin
        self.net, self.input_blob, _ = self.initOpenvinoModel()

    def initOpenvinoModel(self):
        pretrained_models_dir = self.config["MOBILE_FACE_REC"]['Model_dir']
        pretrained_models_name = self.config["MOBILE_FACE_REC"]['Model_name']
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

    def inference(self, img, embed_size=256):
        w, h, c = img.shape 
        if (w == 0) or (h == 0):
            return np.zeros(embed_size)
        frame_input = cv2.resize(img, (128, 128))
        frame_input = frame_input.swapaxes(1, 2).swapaxes(0, 1)
        frame_input = np.reshape(frame_input, [1, 3, 128, 128])
        embed = self.net.infer(inputs={self.input_blob: frame_input})['658']
        return np.squeeze(embed)
