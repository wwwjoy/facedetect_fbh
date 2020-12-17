# -*- coding: utf-8 -*-
"""
Function: Detect
Author: Wujia
Create Time: 2020/8/18 13:49
"""
from FaceDetector import FaceDetector
import glob, os
import cv2
import numpy as np

def draw(img, pred_bboxes, pred_scores, pred_keypoints):
    for box, score, keypoints in zip(pred_bboxes, pred_scores, pred_keypoints):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(img, '%.2f' % score, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        keypoints = np.reshape(keypoints, (-1, 2))

        cv2.circle(img, (int(keypoints[0][0]), int(keypoints[0][1])), 2, (0, 0, 255), 2)
        cv2.circle(img, (int(keypoints[1][0]), int(keypoints[1][1])), 2, (0, 255, 0), 2)
        cv2.circle(img, (int(keypoints[2][0]), int(keypoints[2][1])), 2, (255, 255, 0), 2)
        cv2.circle(img, (int(keypoints[3][0]), int(keypoints[3][1])), 2, (255, 0, 0), 2)
        cv2.circle(img, (int(keypoints[4][0]), int(keypoints[4][1])), 2, (255, 0, 255), 2)

    return img

if __name__ == '__main__':
    detector = FaceDetector(model_path='./models/resnet18.pth', gpu_ids=0, layers=18)  # model_204904   model_262215
    img_paths = glob.glob('./images/*.jpg')
    for img_path in img_paths:
        image = cv2.imread(img_path)
        boxes, landms, scores = detector.detect(image)
        print(boxes, landms, scores)
        d_image = draw(image, boxes, scores, landms)
        cv2.imwrite('./d_images/' + os.path.basename(img_path), d_image)