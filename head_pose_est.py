"""
Openvino head pose estimation model
Intel CNN model, 3D angle regression
"""
import cv2
import os
import numpy as np
import configparser

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin


class HeadPoseEst(object):
    def __init__(self, plugin, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.plugin = plugin
        self.fd_net, self.fd_input_blob, _ = self.initOpenvinoModel()

    def initOpenvinoModel(self):
        pretrained_models_dir = self.config["HEAD_POSE"]['Model_dir']
        pretrained_models_name = self.config["HEAD_POSE"]['Model_name']
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
        h, w, c = img.shape
        finput = cv2.resize(img, (60, 60))
        finput = finput.swapaxes(1, 2).swapaxes(0, 1)
        finput = np.expand_dims(finput, axis=0)
        out = self.fd_net.infer(inputs={self.fd_input_blob: finput})
        y = np.squeeze(out['angle_y_fc'])
        p = np.squeeze(out['angle_p_fc'])
        r = np.squeeze(out['angle_r_fc'])
        return [y, p, r]
