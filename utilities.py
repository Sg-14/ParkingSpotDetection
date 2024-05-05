
import pickle
import numpy as np
from skimage.transform import  resize
import cv2

MODEL = pickle.load(open('model.p', 'rb'))
EMPTY = True
NOT_EMPTY = False

def checkBlockStatus(spot_background):
    flattened = []
    img_resized = resize(spot_background, (15,15,3))
    flattened.append(img_resized.flatten())
    flattened = np.array(flattened)

    result = MODEL.predict(flattened)

    if result == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots(connected_components):

    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1

    for i in range(1, totalLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT]*coef)
        y1 = int(values[i, cv2.CC_STAT_TOP]*coef)
        w = int(values[i, cv2.CC_STAT_WIDTH]*coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT]*coef)

        slots.append([x1, y1, w, h])

    return slots