import cv2
import time
import numpy as np
from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IEPlugin

def initModel(model_xml, model_bin):
    # cpu_ext = "/opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libcpu_extension.so"
    # device = "CPU"
    device = "MYRIAD"
    plugin = IEPlugin(device, plugin_dirs=None)
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net)
    del net
    return exec_net, input_blob, output_blob

def face_embed_lite(face_img, net, input_blob, embed_size=256):
    w, h, c = face_img.shape 
    if (w == 0) or (h == 0):
        return np.zeros(embed_size)
    frame_input = cv2.resize(face_img, (128, 128))
    frame_input = frame_input.swapaxes(1, 2).swapaxes(0, 1)
    frame_input = np.reshape(frame_input, [1, 3, 128, 128])
    frame_input = np.array(frame_input, dtype=np.float32)
    embed = net.infer(inputs={input_blob: frame_input})['658']
    return np.squeeze(embed)

# # s = time.time()
# model_xml = "models/face-reidentification-retail-0095.xml"
# model_bin = "models/face-reidentification-retail-0095.bin"
# net, input_blob, _ = initModel(model_xml, model_bin)
# # print("load model:", time.time() - s)
# img = cv2.imread("face_test.png")
# print(face_embed_lite(img, net, input_blob))