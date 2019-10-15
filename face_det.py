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

def face_detect(img, net, input_blob, conf_threshold):
    h, w, c = img.shape
    finput = cv2.resize(img, (672, 384))
    finput = finput.swapaxes(1, 2).swapaxes(0, 1)
    finput = np.expand_dims(finput, axis=0)
    out = net.infer(inputs={input_blob: finput})
    det_out = out['detection_out'].reshape(-1, 7)
    ret_bbs = []
    for detection in det_out:
        confidence = float(detection[2])
        xmin = int(detection[3] * w)
        ymin = int(detection[4] * h)
        xmax = int(detection[5] * w)
        ymax = int(detection[6] * h)
        if confidence > conf_threshold:
            ret_bbs.append([xmin, ymin, xmax, ymax])
    return ret_bbs


def face_detect_mobile(img, net, input_blob, conf_threshold):
    h, w, c = img.shape
    finput = cv2.resize(img, (300, 300))
    finput = finput.swapaxes(1, 2).swapaxes(0, 1)
    finput = np.expand_dims(finput, axis=0)
    out = net.infer(inputs={input_blob: finput})
    det_out = out['527'].reshape(-1, 7)
    ret_bbs = []
    for detection in det_out:
        confidence = float(detection[2])
        xmin = max(int(detection[3] * w), 0)
        ymin = max(int(detection[4] * h), 0)
        xmax = min(int(detection[5] * w), w)
        ymax = min(int(detection[6] * h), h)
        if confidence > conf_threshold:
            ret_bbs.append([xmin, ymin, xmax, ymax])
    return ret_bbs

# s = time.time()
# model_xml = "/home/pi/models/face-detection-adas-0001.xml"
# model_bin = "/home/pi/models/face-detection-adas-0001.bin"
# net, input_blob, _ = initModel(model_xml, model_bin)
# # model_xml = "/home/pi/models/face-detection-retail-0005.xml"
# # model_bin = "/home/pi/models/face-detection-retail-0005.bin"
# # mobile_net, mobile_input_blob, _ = initModel(model_xml, model_bin)
# print("load models:", time.time() - s)
# img = cv2.imread("test.png")
# confidence = 0.5
# s = time.time()
# out = face_detect(img, net, input_blob, confidence)
# print("model 1 inf.:", time.time() - s)
# # s = time.time()
# # out = face_detect_mobile(img, mobile_net, mobile_input_blob, confidence)
# # print("model 2 inf.:", time.time() - s)
# print(out)
# o = out[0]
# cv2.rectangle(img, (o[0], o[1]), (o[2], o[3]), (255, 0, 0))
# cv2.imwrite("out.png", img)
