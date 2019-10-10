"""
Openvino face detection model
mobilenet backbone - SSD model
"""
import cv2
import os
import numpy as np
import configparser

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin


class MobileFaceDetect(object):
    def __init__(self, plugin, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.plugin = plugin
        self.fd_net, self.fd_input_blob, _ = self.initOpenvinoModel()
        self.confidence = float(self.config["MOBILE_FACE_DET"]['Confidence'])

    def initOpenvinoModel(self):
        pretrained_models_dir = self.config["MOBILE_FACE_DET"]['Model_dir']
        pretrained_models_name = self.config["MOBILE_FACE_DET"]['Model_name']
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
        finput = cv2.resize(img, (300, 300))
        finput = finput.swapaxes(1, 2).swapaxes(0, 1)
        finput = np.expand_dims(finput, axis=0)
        out = self.fd_net.infer(inputs={self.fd_input_blob: finput})
        det_out = out['527'].reshape(-1, 7)
        ret_bbs = []
        for detection in det_out:
            confidence = float(detection[2])
            xmin = max(int(detection[3] * w), 0)
            ymin = max(int(detection[4] * h), 0)
            xmax = min(int(detection[5] * w), w)
            ymax = min(int(detection[6] * h), h)
            if confidence > self.confidence:
                ret_bbs.append([xmin, ymin, xmax, ymax])
        return ret_bbs
