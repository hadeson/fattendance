"""
Openvino face embedding model
"""
import cv2
import os
import numpy as np
import configparser

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin


class FaceEmbedding(object):
    def __init__(self, plugin, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.plugin = plugin
        self.fr_net, self.fr_input_blob, _ = self.initOpenvinoModel()
        self.confidence = float(self.config["ARCFACE"]['Confidence'])
        self.embed_size = int(self.config["ARCFACE"]['Embed_size'])

    def initOpenvinoModel(self):
        pretrained_models_dir = self.config["ARCFACE"]['Model_dir']
        pretrained_models_name = self.config["ARCFACE"]['Model_name']
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

    def inference(self, face_img, tta=True):
        """
        IR-50_A model
        input 128x128
        output 512
        """
        # resize image to [128, 128]
        # resized = cv2.resize(face_img, (128, 128))

        # center crop image
        # a = int((128-112)/2)  # x start
        # b = int((128-112)/2+112)  # x end
        # c = int((128-112)/2)  # y start
        # d = int((128-112)/2+112)  # y end
        # ccropped = resized[a:b, c:d]  # center crop the image
        resized = cv2.resize(face_img, (112, 112))
        ccropped = resized[..., ::-1]  # BGR to RGB

        # flip image horizontally
        flipped = cv2.flip(ccropped, 1)

        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype=np.float32)
        ccropped = (ccropped - 127.5) / 128.0

        if tta:
            flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
            flipped = np.reshape(flipped, [1, 3, 112, 112])
            flipped = np.array(flipped, dtype=np.float32)
            flipped = (flipped - 127.5) / 128.0

            # extract features
            crop_output = self.fr_net.infer(inputs={self.fr_input_blob: ccropped})['536']
            flip_output = self.fr_net.infer(inputs={self.fr_input_blob: flipped})['536']
            emb_batch = crop_output + flip_output
            features = self.l2_norm_numpy(emb_batch)
        else:
            crop_output = self.fr_net.infer(inputs={self.fr_input_blob: ccropped})['536']
            features = self.l2_norm_numpy(crop_output)
        return features

    def l2_norm_numpy(self, input):
        norm = np.linalg.norm(input)
        output = input / norm
        return output
