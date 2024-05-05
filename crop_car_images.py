
import os
import cv2

output_dir = './parking-image-data'
mask_path = 'ParkingLotVideo/mask_1920_1080.png'
mask = cv2.imread(mask_path, 0)

analysis = cv2.connectedComponentsWithStats(mask,4, cv2.CV_32S)

(totalLabels, label_ids, values, centroid) = analysis

slots = []

for i in range(1, totalLabels):

    area = values[i, cv2.CC_STAT_AREA]

    #Extracting the coordinates
    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]

    #Coordinates of the boxes
    pt1 = (x1, y1)
    pt2 = (x1+w, y1+h)
    (X, Y) = centroid(i)

    slots.append([x1, y1, w, h])

video_path = 'ParkingLotVideo/parking_1920_1080.mp4'

video = cv2.VideoCapture(video_path)

frame_nmr = 0
video.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
ret, frame = video.read()

while ret:

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = video.read()

    if ret:
        for slot_nmr, slot in enumerate(slots):
            if slot_nmr in [132, 147, 164, 180, 344, 360, 377, 385,
                            341, 360, 179, 131, 106, 91, 61, 4, 89,
                            129, 161, 185, 201, 224, 271, 303, 319,
                            335, 351, 389, 29, 12, 32, 72, 281, 280,
                            157, 223, 26]:
                slot = frame[slot[1]:slot[1]+slot[3],slot[0]:slot[0]+slot[2], :]
                cv2.imwrite(os.path.join(output_dir, '{}_{}.jpg'.format(
                    str(frame_nmr).zfill(8), str(slot_nmr).zfill(8)
                )), slot)

        frame_nmr += 10

video.release()