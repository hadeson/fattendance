"""
face registration
"""
import cv2
import time
import os
import pickle
import numpy as np
from face_det import face_detect_mobile as face_detect
# from face_rec import face_embed_lite as face_embed
from arcface import FaceEmbedding
from face_search import bruteforce
from utils import initModel, area, loadData
from openvino.inference_engine import IENetwork, IEPlugin

def main():
    cam = cv2.VideoCapture(0)
    # load data
    data_dir= "data"
    # embed_file_path = "data/embed.pkl"
    embed_file = os.path.join(data_dir, "embed.pkl")
    face_database = loadData(embed_file)
    # load models
    fe_model_xml = "models/face-reidentification-retail-0095.xml"
    fe_model_bin = "models/face-reidentification-retail-0095.bin"
    fd_model_xml = "models/face-detection-retail-0005.xml"
    fd_model_bin = "models/face-detection-retail-0005.bin"
    device = "MYRIAD"
    plugin = IEPlugin(device, plugin_dirs=None)
    fd_net, fd_input_blob, _ = initModel(fd_model_xml, fd_model_bin, plugin)
    # fe_net, fe_input_blob, _ = initModel(fe_model_xml, fe_model_bin, plugin)
    face_embed = FaceEmbedding(plugin)
    # params
    fd_conf = 0.5
    fm_threshold = 0.6
    label = "new"
    period = 1
    button_pressed = False
    max_num = 3
    num = 0
    face_features = []
    face_imgs = []

    s = time.time()
    while(True):
        ret, frame = cam.read()
        if not ret:
            print("dead cam")
            cam = cv2.VideoCapture(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        face_bboxes = []
        # if time.time() - s > period:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            button_pressed = True
        if button_pressed and (num < max_num):
            face_bboxes = face_detect(frame, fd_net, fd_input_blob, fd_conf)
        if (len(face_bboxes) > 0) and button_pressed:
            areas = [area(box) for box in face_bboxes]
            max_id = np.argmax(np.asarray(areas))
            mfb = face_bboxes[max_id]
            main_face = frame[mfb[1]:mfb[3], mfb[0]:mfb[2], :]
            # TODO real face detection
            # TODO face alignment
            face_feature = face_embed.inference(main_face)
            # best_match = bruteforce(face_feature, face_database, fm_threshold)
            # if best_match is None:
            #     label = "new"
            # else:
            #     label = str(best_match['id'])
            face_features.append(face_feature)
            face_imgs.append(main_face)
            num += 1
            button_pressed = False
            s = time.time()
            # visualize for debug
            cv2.rectangle(frame, (mfb[0], mfb[1]), (mfb[2], mfb[3]), (255, 0, 0), 2)
            cv2.putText(frame, str(num), (mfb[0], mfb[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), lineType=cv2.LINE_AA)
            print(num)

        if num >= max_num:
            new_id = len(face_database)
            new_face = {
                'id': new_id,
                'feats': face_features
            }
            face_database.append(new_face)
            pickle.dump(face_database, open(embed_file, "wb"))
            img_dir = os.path.join(data_dir, str(new_id))
            os.mkdir(img_dir)
            for i, face in enumerate(face_imgs):
                img_path = os.path.join(img_dir, "{}.jpg".format(i))
                cv2.imwrite(img_path, face)
            face_imgs = []
            face_features = []
            num = 0
            s = time.time()
            print("done!")

        cv2.imshow("face registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

if __name__ == '__main__':
    main()