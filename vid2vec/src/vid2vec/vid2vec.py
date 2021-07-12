# -*- coding: utf-8 -*-
"""
Research code from
https://colab.research.google.com/drive/1PJDTfLYwRRzxhgW3yIe_OymnkDKKG5kB

2D use - face_recognition package
3D use - face_alignment


"""
# from multiprocessing import Process
import os
import argparse
import sys
import os
import time
import logging

import face_alignment  # for 3D face landmarks
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv
from mlr import Face

import traceback

from vid2vec import __version__

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"

_logger = logging.getLogger(__name__)


# Global configuration
lip_features = []
broken_frame_count = 0
fa = None
# mode2d = False

# getMouthImage (from TLR Teeth Appearance Calculation.ipynb)


def getMouthImage(faceImage, face_landmarks=None, margin=0):
    if face_landmarks == None:
        # face_locations = face_recognition.face_locations(faceImage,model='cnn')[0]
        # face_landmarks = face_recognition.face_landmarks(faceImage,face_locations=[face_locations])[0]  # Assume first face
        face_landmarks = face_recognition.face_landmarks(
            faceImage)[0]  # Assume first face

    minx = miny = float('inf')
    maxx = maxy = float('-inf')

    for x, y in face_landmarks['top_lip']:
        minx = min(minx, x)
        miny = min(miny, y)

    for x, y in face_landmarks['bottom_lip']:
        maxx = max(maxx, x)
        maxy = max(maxy, y)

    mouthImage = faceImage[miny-margin:maxy+margin, minx-margin:maxx+margin]

    # lip_landmarks must be translate to origin (0,0) by minx, miny

    lip_landmarks = {
        'top_lip': [],
        'bottom_lip': []
    }

    for p in face_landmarks['top_lip']:
        p2 = (p[0] - minx, p[1] - miny)
        lip_landmarks['top_lip'].append(p2)

    for p in face_landmarks['bottom_lip']:
        p2 = (p[0] - minx, p[1] - miny)
        lip_landmarks['bottom_lip'].append(p2)

    return mouthImage, lip_landmarks


def getMouthImage_3D(faceImage, face_landmarks, margin=0):
    minx = miny = float('inf')
    maxx = maxy = float('-inf')

    # print(f"\t{face_landmarks}")
    lip_landmarks = face_landmarks[48:69]
    for x, y, z in lip_landmarks:
        # print(x,y)
        minx = int(min(minx, x))
        miny = int(min(miny, y))
        maxx = int(max(maxx, x))
        maxy = int(max(maxy, y))

    # print("------------- ",(minx,miny),(maxx,maxy))
    # print("------------- ",(miny-margin,maxy+margin),(minx-margin,maxx+margin))
    mouthImage = faceImage[miny-margin:maxy+margin, minx-margin:maxx+margin]

    return mouthImage, lip_landmarks

# Ray tracing (from TLR Teeth Appearance Calculation.ipynb)


def ray_tracing_method(x, y, poly):

    n = len(poly)
    inside = False

    p1x, p1y, p1z = poly[0]
    for i in range(n+1):
        p2x, p2y, p1z = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def isin_inner_mouth(lip_boundary, x, y):
    """
    isin_inner_mouth (from TLR Teeth Appearance Calculation.ipynb)
    """
    bounds = lip_boundary
    isin = ray_tracing_method(x, y, bounds)
    return isin

    # face_recognition
    top_lip = lip_boundary['top_lip']
    bottom_lip = lip_boundary['bottom_lip']
    bounds = np.concatenate((top_lip[6:], bottom_lip[6:]), axis=0)
    isin = ray_tracing_method(x, y, bounds)
    return isin


# findCavity (from TLR Teeth Appearance Calculation.ipynb)


def findCavity(top_lip, bottom_lip):
    return np.concatenate((top_lip[6:], bottom_lip[6:]), axis=0)

# cavityArea (from TLR Teeth Appearance Calculation.ipynb)


def cavityArea(top_lip, bottom_lip):
    cavity = findCavity(top_lip, bottom_lip)
#   cavity = np.concatenate((top_lip[6:], bottom_lip[6:]),axis=0)
    x = cavity[:, 0]
    y = cavity[:, 1]
    return PolyArea(x, y)

# getTeethScore (from TLR Teeth Appearance Calculation.ipynb)


def getTeethScore(mouthImage, lip_landmarks):

    height, width, channels = mouthImage.shape

    area = height * width

    # Operate in BGR (imread loads in BGR)
    # OR WHAT???
    # Working with VDO frame
    # - RGB2Lab gives all WHITE region
    lab = cv2.cvtColor(mouthImage, cv2.COLOR_RGB2Lab)
    luv = cv2.cvtColor(mouthImage, cv2.COLOR_RGB2Luv)
#   lab = cv2.cvtColor(mouthImage, cv2.COLOR_BGR2Lab)
#   luv = cv2.cvtColor(mouthImage, cv2.COLOR_BGR2Luv)

    lab_ud = lab[:, :, 1].mean() - lab[:, :, 1].std()
    ta = lab_ud  # From THESIS (LAB, LUV)

    luv_ud = luv[:, :, 1].mean() - luv[:, :, 1].std()
    tu = luv_ud  # from thesis

    # WHY do we copy?
    lab2 = np.copy(lab)
    luv2 = np.copy(luv)

    # Copy for teeth hilight
    hilightedMouthImage = np.copy(mouthImage)

    # Pixel-wise operation
    # TODO make it faster?
    lab_c = luv_c = 0  # Counters
    for y in range(len(hilightedMouthImage)):
        row = hilightedMouthImage[y]
        for x in range(len(row)):
            inMouth = isin_inner_mouth(lip_landmarks, x, y)

            if inMouth:
                p = row[x]
                lab_a = lab2[y, x, 1]
                luv_a = luv2[y, x, 1]
                if lab_a <= ta:
                    p[0] = 255  # L
                    p[1] = 255  # L
                    p[2] = 255  # L
                    lab_c += 1
                if luv_a <= tu:
                    p[0] = 255  # L
                    p[1] = 255  # L
                    p[2] = 255  # L
                    luv_c += 1

    return (hilightedMouthImage, lab, luv, lab_c, luv_c)


def getTeethScore_3D(mouthImage, lip_landmarks=None):

    height, width, channels = mouthImage.shape

    area = height * width

    # Operate in BGR (imread loads in BGR)
    # OR WHAT???
    # Working with VDO frame
    # - RGB2Lab gives all WHITE region
    lab = cv2.cvtColor(mouthImage, cv2.COLOR_RGB2Lab)
    luv = cv2.cvtColor(mouthImage, cv2.COLOR_RGB2Luv)
#   lab = cv2.cvtColor(mouthImage, cv2.COLOR_BGR2Lab)
#   luv = cv2.cvtColor(mouthImage, cv2.COLOR_BGR2Luv)

    lab_ud = lab[:, :, 1].mean() - lab[:, :, 1].std()
    ta = lab_ud  # From THESIS (LAB, LUV)

    luv_ud = luv[:, :, 1].mean() - luv[:, :, 1].std()
    tu = luv_ud  # from thesis

    # WHY do we copy?
    lab2 = np.copy(lab)
    luv2 = np.copy(luv)

    # Copy for teeth hilight
    hilightedMouthImage = np.copy(mouthImage)

    # Pixel-wise operation
    # TODO make it faster?
    lab_c = luv_c = 0  # Counters
    for y in range(len(hilightedMouthImage)):
        row = hilightedMouthImage[y]
        for x in range(len(row)):
            inMouth = False
            if lip_landmarks is None:
                inMouth = isin_mouth(hilightedMouthImage, x, y)
            else:
                inMouth = isin_inner_mouth(lip_landmarks, x, y)

            if inMouth:
                p = row[x]
                lab_a = lab2[y, x, 1]
                luv_a = luv2[y, x, 1]
                if lab_a <= ta:
                    p[0] = 255  # L
                    p[1] = 255  # L
                    p[2] = 255  # L
                    lab_c += 1
                if luv_a <= tu:
                    p[0] = 255  # L
                    p[1] = 255  # L
                    p[2] = 255  # L
                    luv_c += 1

    return (hilightedMouthImage, lab, luv, lab_c, luv_c)


# draw_bounary
def draw_boundary(facial_feature):
    # _logger.debug(type(face_landmarks[facial_feature]),face_landmarks[facial_feature])
    points = face_landmarks[facial_feature]

    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))

    cv2.polylines(frame, points, True, (255, 255, 255), thickness=4)


def flat_dict(fl, frame_number, lab_c, luv_c):
    fl2 = {}
    fl2["frame#"] = frame_number
    point_no = 1
    for p in fl:
        keyx = f"{point_no}_x"
        keyy = f"{point_no}_y"
        keyz = f"{point_no}_z"

        fl2[keyx] = p[0]
        fl2[keyy] = p[1]
        fl2[keyz] = p[2]
        point_no += 1

    fl2["teeth_LAB"] = lab_c
    fl2["teeth_LUV"] = luv_c
    return fl2

    # OLD CODE for face_recognition
    point_no = 1
    for part_name in fl.keys():  # ex: chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip
        # print(part_name)
        part = fl[part_name]
        for point in part:
            fl2_key_x = "%d_%s_x" % (point_no, part_name)
            fl2_key_y = "%d_%s_y" % (point_no, part_name)
            fl2[fl2_key_x] = point[0]
            fl2[fl2_key_y] = point[1]
            point_no += 1

    fl2["teeth_LAB"] = lab_c
    fl2["teeth_LUV"] = luv_c

    return fl2


def compute_features(frame_number, frame):
    global lip_features
    # from skimage import io
    # import matplotlib.pyplot as plt

    # step_marker = 10
    elapsed_time = 0
    print(f"compute_features_3D: {frame_number}")
    markedMouthImage = None
    try:
        start_time = time.time()
        print("\tfa.get_landmarks")
        preds = fa.get_landmarks(frame)
        face_landmarks = preds[0]
        f = Face.Face(frame,face_landmarks)

        print("\t#Face Found: ", len(preds))
        # face_landmarks_list = face_recognition.face_landmarks(frame)
        # face_landmarks = preds[0]  # assume first face found
        # face_landmarks = face_landmarks_list
        # print("face_landmarks.shape", face_landmarks.shape)

        # print("\tgetMouthImage_3D")
        # mouthImage, lip_landmarks = getMouthImage_3D(frame, face_landmarks)
        
        print("\tgetTeethScore_3D")
        # score = getTeethScore_3D(mouthImage, lip_landmarks)
        (markedMouthImage, lab, luv, lab_c, luv_c,tr_lab,tr_luv) = f.getTeethScore()
        print(f"LAB_C {lab_c}\nLUV_C {luv_c}")

        # print("flat_dict")
        fl2 = flat_dict(face_landmarks, frame_number, lab_c, luv_c)

        # print(face_landmarks)
        # print(fl2)
        # lip_features.append({
        #     "frame_id": frame_number,
        #     "top_lip": face_landmarks['top_lip'],
        #     "bottom_lip": face_landmarks['bottom_lip'],
        #     "teeth_appearance": {
        #         "LAB": lab_c,
        #         "LUV": luv_c
        #     }
        # })
        lip_features.append(fl2)
        # print(lip_features)

        # print("#", end='')
        # if frame_number % step_marker == 0:
        #     print(" %d/%d" % (frame_number, length))

        end_time = time.time()
        elapsed_time = end_time-start_time

    except Exception as e:
        _logger.error(e)
        # return (0, None, None)
        tb = traceback.format_exc()
        print(tb)
        exit

    return (elapsed_time, preds, markedMouthImage)


def ___compute_features_2D(frame_number, frame):
    global lip_features
    # step_marker = 10
    try:
        start_time = time.time()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        face_landmarks = face_landmarks_list[0]  # assume first face found
        mouthImage, lip_landmarks = getMouthImage(
            frame, face_landmarks=face_landmarks)
        score = getTeethScore(mouthImage, lip_landmarks)
        markedMouthImage = score[0]
        lab_c = score[3]
        luv_c = score[4]

        # print(face_landmarks)
        fl2 = flat_dict(face_landmarks, frame_number, lab_c, luv_c)
        # print(fl2)

        # lip_features.append({
        #     "frame_id": frame_number,
        #     "top_lip": face_landmarks['top_lip'],
        #     "bottom_lip": face_landmarks['bottom_lip'],
        #     "teeth_appearance": {
        #         "LAB": lab_c,
        #         "LUV": luv_c
        #     }
        # })
        lip_features.append(fl2)
        # print(lip_features)

        # print("#", end='')
        # if frame_number % step_marker == 0:
        #     print(" %d/%d" % (frame_number, length))

        end_time = time.time()
        elapsed_time = end_time-start_time
    except Exception as e:
        _logger.error(e)
        return (0, None, None)
    return (elapsed_time, face_landmarks_list, markedMouthImage)

# extract_lips

def extract_features(ifn, skip_frames=0, write_output_movie=False):
    global lip_features
    global broken_frame_count
    global fa
    lip_features.clear()
    _logger.debug("Input File: {}".format(ifn))
    # It only works with AVI
    ofn = ifn.replace(".mp4", "")+"-skip-{}-output.avi".format(skip_frames)

    input_movie = cv2.VideoCapture(ifn)
    if not input_movie.isOpened():
        _logger.debug("could not open :", ifn)

    frame_count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_movie.get(cv2.CAP_PROP_FPS)
    codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))

    _logger.debug(input_movie)
    _logger.debug("CODEC: {}".format(codec))
    _logger.debug("FPS: {}".format(fps))
    _logger.debug("Dimension: {}x{}".format(frame_width, frame_height))
    _logger.debug("Length: {}".format(frame_count))
    _logger.debug("SKIP: {}".format(skip_frames))

    expected_lip_features_size = int(frame_count / (skip_frames+1))
    step_marker = int(frame_count / 10)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    if write_output_movie:
        output_movie = cv2.VideoWriter(
            ofn, fourcc, fps, (frame_width, frame_height))
        _logger.debug("OUTPUT MOVIE: {}".format(output_movie))
    # output_movie.release()

    frame_number = 0
    frame = None
    observe_frame = 100
    total_time = 0
    # processes = []
    start_time = time.time()

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False,device='cuda')
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')

    while True:

        frame_number += 1
        ret, frame = input_movie.read()  # Grab a single frame of video

        # Read frame and throw away, otherwise it won't stop with frameskip condition.
        if frame_number % (skip_frames + 1) != 0:
            continue

        # Quit when the input video file ends
        if not ret:
            break

        _logger.debug(f"compute_features {frame_number}/{frame_count}")

        # This adds to global var, lip_features
        (et, face_landmarks_list, markedMouthImage) = compute_features(frame_number, frame)

        if face_landmarks_list is None:
            broken_frame_count += 1
            continue

        _logger.debug(
            f"\n\tVDO File: {ifn}\n\t...features size: {len(lip_features)}/{expected_lip_features_size}\n\t...Elapsed: {et}")

    # Save to CSV
    output_csv_filename = ifn.replace(".mp4", "").replace(".avi", "") + ".csv"
    _logger.info(f"Writing to {output_csv_filename}")
    print(f"Writing to {output_csv_filename}")
    row_no = 0
    with open(output_csv_filename, "w", newline='') as f:
        csv_writer = csv.writer(f)
        for r in lip_features:
            if row_no == 0:
                # Writing headers of CSV file
                header = r.keys()
                csv_writer.writerow(header)
                row_no += 1

            # Writing data of CSV file
            csv_writer.writerow(r.values())
            # _logger.debug(r.values())
    _logger.info(f".... Wrote {row_no} records  Done")
    
def __extract_features(ifn, skip_frames=0, write_output_movie=False):
    global lip_features
    global broken_frame_count
    lip_features.clear()
    _logger.debug("Input File: {}".format(ifn))
    # It only works with AVI
    ofn = ifn.replace(".mp4", "")+"-skip-{}-output.avi".format(skip_frames)

    input_movie = cv2.VideoCapture(ifn)
    if not input_movie.isOpened():
        _logger.debug("could not open :", ifn)

    frame_count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_movie.get(cv2.CAP_PROP_FPS)
    codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))

    _logger.debug(input_movie)
    _logger.debug("CODEC: {}".format(codec))
    _logger.debug("FPS: {}".format(fps))
    _logger.debug("Dimension: {}x{}".format(frame_width, frame_height))
    _logger.debug("Length: {}".format(frame_count))
    _logger.debug("SKIP: {}".format(skip_frames))

    expected_lip_features_size = int(frame_count / (skip_frames+1))
    step_marker = int(frame_count / 10)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # output_movie = cv2.VideoWriter(ofn, codec, fps, (frame_width, frame_height))

    if write_output_movie:
        output_movie = cv2.VideoWriter(
            ofn, fourcc, fps, (frame_width, frame_height))
        _logger.debug("OUTPUT MOVIE: {}".format(output_movie))
    # output_movie.release()

    # Initialize variables
    # face_locations = []
    frame_number = 0

    # lip_features = []
    frame = None
    observe_frame = 100
    total_time = 0
    # processes = []
    start_time = time.time()
    while True:

        frame_number += 1

        ret, frame = input_movie.read()  # Grab a single frame of video

        # Read frame and throw away, otherwise it won't stop with frameskip condition.
        if frame_number % (skip_frames + 1) != 0:
            continue

        # Quit when the input video file ends
        if not ret:
            break

        _logger.debug(f"compute_features {frame_number}/{frame_count}")

        (et, face_landmarks_list, markedMouthImage) = compute_features(frame_number, frame)

        if face_landmarks_list is None:
            broken_frame_count += 1
            continue

        _logger.debug(
            f"\n\tVDO File: {ifn}\n\t...features size: {len(lip_features)}/{expected_lip_features_size}\n\t...Elapsed: {et}")

    # Save to CSV
    output_csv_filename = ifn.replace(".mp4", "").replace(".avi", "") + ".csv"
    _logger.info(f"Writing to {output_csv_filename}")
    row_no = 0
    with open(output_csv_filename, "w", newline='') as f:
        csv_writer = csv.writer(f)
        for r in lip_features:
            if row_no == 0:
                # Writing headers of CSV file
                header = r.keys()
                csv_writer.writerow(header)
                row_no += 1

            # Writing data of CSV file
            csv_writer.writerow(r.values())
            # _logger.debug(r.values())
    _logger.info(f".... Wrote {row_no} records  Done")


def ____extract_features_2D(ifn, skip_frames=0, write_output_movie=False):
    global lip_features
    global broken_frame_count
    lip_features.clear()
    _logger.debug("Input File: {}".format(ifn))
    # It only works with AVI
    ofn = ifn.replace(".mp4", "")+"-skip-{}-output.avi".format(skip_frames)

    input_movie = cv2.VideoCapture(ifn)
    if not input_movie.isOpened():
        _logger.debug("could not open :", ifn)

    frame_count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_movie.get(cv2.CAP_PROP_FPS)
    codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))

    _logger.debug(input_movie)
    _logger.debug("CODEC: {}".format(codec))
    _logger.debug("FPS: {}".format(fps))
    _logger.debug("Dimension: {}x{}".format(frame_width, frame_height))
    _logger.debug("Length: {}".format(frame_count))
    _logger.debug("SKIP: {}".format(skip_frames))

    expected_lip_features_size = int(frame_count / (skip_frames+1))
    step_marker = int(frame_count / 10)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # output_movie = cv2.VideoWriter(ofn, codec, fps, (frame_width, frame_height))

    if write_output_movie:
        output_movie = cv2.VideoWriter(
            ofn, fourcc, fps, (frame_width, frame_height))
        _logger.debug("OUTPUT MOVIE: {}".format(output_movie))
    # output_movie.release()

    # Initialize variables
    # face_locations = []
    frame_number = 0

    # lip_features = []
    frame = None
    observe_frame = 100
    total_time = 0
    # processes = []
    start_time = time.time()
    while True:

        frame_number += 1

        ret, frame = input_movie.read()  # Grab a single frame of video

        # Read frame and throw away, otherwise it won't stop with frameskip condition.
        if frame_number % (skip_frames + 1) != 0:
            continue

        # end_time = time.time()
        # _logger.debug("\tLoad frame {}: {}".format(frame_number,(end_time - start_time)))

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        # face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

        # TODO ..  make it multithread :(
        """
    p = Process(target=compute_features, args=(frame_number,frame,))
    processes.append(p)
    p.start()
    """
        _logger.debug(f"compute_features {frame_number}/{frame_count}")
        (et, face_landmarks_list, markedMouthImage) = compute_features_2D(
            frame_number, frame)

        if face_landmarks_list == None:
            broken_frame_count += 1
            continue

        _logger.debug(
            f"\n\tVDO File: {ifn}\n\t...features size: {len(lip_features)}/{expected_lip_features_size}\n\t...Elapsed: {et}")


#################################
#     face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

#     # Save lip features locations
#     # Assume only single face
#     face_landmarks = face_landmarks_list[0]


#     mouthImage,lip_landmarks = getMouthImage(rgb_frame)
#     score = getTeethScore(mouthImage,lip_landmarks)
# #     _logger.debug('LAB {}\nLUV {}'.format(score[3],score[4]))

#     markedMouthImage = score[0]
#     lab_c = score[3]
#     luv_c = score[4]

#     lip_features.append({
#         "frame_id": frame_number,
#         "top_lip": face_landmarks_list[0]['top_lip'],
#         "bottom_lip": face_landmarks_list[0]['bottom_lip'],
#         "teeth_appearance": {
#             "LAB": lab_c,
#             "LUV": luv_c
#         }
#     })
######################
        # Let's trace out each facial feature in the image with a line!
    #     for facial_feature in face_landmarks.keys():
    #       draw_boundary(facial_feature)

        # Write the resulting image to the output video file
    #     _logger.debug("Writing frame {} / {}".format(frame_number, length))
        # print("#",end='')
        # if frame_number % step_marker == 0:
        #     _logger.debug(" %d/%d" % (frame_number, length))

        if write_output_movie:
            #     i/len(some_list)*100," percent complete         \r",
            # Drawing mouth image on top of the face
            # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
            x_offset = y_offset = float('inf')
            for x, y in face_landmarks_list[0]['top_lip']:
                x_offset = min(x_offset, x)
                y_offset = min(y_offset, y)

            markedMouthImage = markedMouthImage[:, :, ::-1]
            frame[y_offset:y_offset+markedMouthImage.shape[0],
                  x_offset:x_offset+markedMouthImage.shape[1]] = markedMouthImage
    #     if frame_number == observe_frame: break
            output_movie.write(frame)
#     output_movie.write(markedMouthImage)
    end_time = time.time()

    # for p in processes:
    #     p.join()

    _logger.debug("Elapse Time: {}".format(end_time - start_time))

    if write_output_movie:
        output_movie.release()
#   plt.imshow(frame[:, :, ::-1])

    # Save to CSV
    output_csv_filename = ifn.replace(".mp4", "").replace(".avi", "") + ".csv"
    _logger.info(f"Writing {row_no} records to {output_csv_filename}")
    with open(output_csv_filename, "w", newline='') as f:
        csv_writer = csv.writer(f)
        row_no = 0
        for r in lip_features:
            if row_no == 0:
                # Writing headers of CSV file
                header = r.keys()
                csv_writer.writerow(header)
                row_no += 1

            # Writing data of CSV file
            csv_writer.writerow(r.values())
            # _logger.debug(r.values())
        _logger.info(f".... Done")

    # import json as j
    # # outputFilename = ifn+".json"
    # # with open(outputFilename,"w") as f:
    # _logger.debug("Saving JSON")
    # with open(ofn, "w") as f:
    #     j.dump(lip_features, f, indent=4)

    # return outputFilename


# def vid2vec(video_filename, skip_frames=0, mode2d=False):
def vid2vec(video_filename, skip_frames=0):
    """Main entry for vid2vec function. will generate a JSON file of property vector of the given video

    Args:
      video_filename (str): Video filename (mp4, avi)
      skip_frames (int): number of frames to skip
    #   mode2d (bool): false for 3D (default), otherwise, true for 2D

    Returns:
      int: -1 video does not exist
    """
    # clips = ['v1.mp4','v2.mp4','v3.mp4']
    # clips = ['v5.mp4'] # The best word separation with pauses. But teeth too dark
    # clips = [v]
    # for c in clips:
    #   extract_lips(c)

    try:
        with open(video_filename) as f:
            # Output to AVI
            # current_dir = os.getcwd()
            # basename = os.path.basename(video_filename)
            # sep = os.path.sep
            # ofn = "{}{}{}.json".format(current_dir, sep, basename)
            # _logger.debug("\tOutput to "+ofn)
            # extract_features(video_filename, skip_frames, false)

            # if mode2d:
            #     extract_features_2D(video_filename, skip_frames, True)
            # else:
            #     extract_features_3D(video_filename, skip_frames, True)
            extract_features(video_filename, skip_frames, True)

            return 0
    except IOError:
        _logger.warn('File "{}" not accessible'.format(video_filename))
        return -1

    return video_filename


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate a CSV vector file of a video clip.")
    parser.add_argument(
        "--version",
        action="version",
        version="vid2vec {ver}".format(ver=__version__))
    parser.add_argument_group()
    parser.add_argument(
        "-s",
        nargs='?',
        type=int,
        default=0,
        help="number of frames to be skipped")
    parser.add_argument(
        dest="v",
        nargs="+",
        help="video filename",
        type=str,
        metavar="FILENAME")
    # parser.add_argument(
    #     "-2d",
    #     dest="mode2d",
    #     help="Use 2D face landmarks",
    #     action="store_const",
    #     const=True)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    global broken_frame_count
    # global mode2d
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    # mode2d = False if args.mode2d is None else True
    # _logger.debug(f'mode2d {mode2d}')

    video_filename = args.v[0]
    _logger.debug("INPUT FILENAME: {}".format(video_filename))
    if not os.path.exists(video_filename):
        print("File does not exist")
        return
    # _logger.debug("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    ret = vid2vec(video_filename, args.s)
    _logger.debug("RET {}".format(ret))
    _logger.debug("Broken Frame Count: {}".format(broken_frame_count))
    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
