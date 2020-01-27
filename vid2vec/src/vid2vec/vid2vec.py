# -*- coding: utf-8 -*-
"""
Research code from
https://colab.research.google.com/drive/1PJDTfLYwRRzxhgW3yIe_OymnkDKKG5kB


This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

  console_scripts =
        fibonacci = vid2vec.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""
from multiprocessing import Process
import os
import argparse
import sys
import os
import time
import logging

import face_recognition
import numpy as np
from matplotlib import pyplot as plt
import cv2

from vid2vec import __version__

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"

_logger = logging.getLogger(__name__)




# getMouthImage (from TLR Teeth Appearance Calculation.ipynb)
def getMouthImage(faceImage,face_landmarks=None,margin=0):
  # face_locations = face_recognition.face_locations(faceImage)
  if face_landmarks == None:
    face_landmarks = face_recognition.face_landmarks(faceImage)[0]  # Assume first face

  
  minx = miny = float('inf')
  maxx = maxy = float('-inf')

  for x,y in face_landmarks['top_lip']:
    minx = min(minx,x)
    miny = min(miny,y)

  for x,y in face_landmarks['bottom_lip']:
    maxx = max(maxx,x)
    maxy = max(maxy,y)

  mouthImage = faceImage[miny-margin:maxy+margin,minx-margin:maxx+margin]
  
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

  return mouthImage,lip_landmarks

# Ray tracing (from TLR Teeth Appearance Calculation.ipynb)
def ray_tracing_method(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside
  
# isin_inner_mouth (from TLR Teeth Appearance Calculation.ipynb)
def isin_inner_mouth(lip_boundary,x,y):
  top_lip = lip_boundary['top_lip']
  bottom_lip = lip_boundary['bottom_lip']
  bounds = np.concatenate((top_lip[6:], bottom_lip[6:]),axis=0)
  isin = ray_tracing_method(x,y,bounds)
  return isin
  
# findCavity (from TLR Teeth Appearance Calculation.ipynb)
def findCavity(top_lip,bottom_lip):
  return np.concatenate((top_lip[6:], bottom_lip[6:]),axis=0)

# cavityArea (from TLR Teeth Appearance Calculation.ipynb)
def cavityArea(top_lip,bottom_lip):
  cavity = findCavity(top_lip,bottom_lip)
#   cavity = np.concatenate((top_lip[6:], bottom_lip[6:]),axis=0)
  x = cavity[:,0]
  y = cavity[:,1]
  return PolyArea(x,y)

# getTeethScore (from TLR Teeth Appearance Calculation.ipynb)
def getTeethScore(mouthImage,lip_landmarks=None):

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


  lab_ud = lab[:,:,1].mean() - lab[:,:,1].std()
  ta = lab_ud # From THESIS (LAB, LUV)
  
    
  luv_ud = luv[:,:,1].mean() - luv[:,:,1].std()
  tu = luv_ud # from thesis
 


  # WHY do we copy?
  lab2 = np.copy(lab)
  luv2 = np.copy(luv)
  
  # Copy for teeth hilight 
  hilightedMouthImage = np.copy(mouthImage)


    
  # Pixel-wise operation
  # TODO make it faster?
  lab_c = luv_c = 0 # Counters
  for y in range(len(hilightedMouthImage)):
    row = hilightedMouthImage[y]
    for x in range(len(row)):
      inMouth = False
      if lip_landmarks == None:
        inMouth = isin_mouth(hilightedMouthImage,x,y)
      else:
        inMouth = isin_inner_mouth(lip_landmarks,x,y)
          
      if inMouth:
        p = row[x]
        lab_a = lab2[y,x,1]
        luv_a = luv2[y,x,1]
        if lab_a <= ta:
          p[0] = 255 # L
          p[1] = 255 # L
          p[2] = 255 # L
          lab_c += 1
        if luv_a <= tu:
          p[0] = 255 # L
          p[1] = 255 # L
          p[2] = 255 # L
          luv_c += 1
          
  return (hilightedMouthImage,lab,luv,lab_c,luv_c)

  
# draw_bounary
def draw_bounary(facial_feature):
  # print(type(face_landmarks[facial_feature]),face_landmarks[facial_feature])
  points = face_landmarks[facial_feature]

  points = np.array(points, np.int32)
  points = points.reshape((-1,1,2))

  cv2.polylines(frame,points,True,(255,255,255),thickness=4)
  

def compute_features(frame_number,frame,lip_features):
  print("compute_features {}".format(frame_number))
  start_time = time.time()
  face_landmarks_list = face_recognition.face_landmarks(frame)
  face_landmarks = face_landmarks_list[0] # assume first face found
  mouthImage,lip_landmarks = getMouthImage(frame,face_landmarks=face_landmarks)
  score = getTeethScore(mouthImage,lip_landmarks)
  # markedMouthImage = score[0]
  lab_c = score[3]
  luv_c = score[4]
  
  lip_features.append({
      "frame_id": frame_number,
      "top_lip": face_landmarks['top_lip'],
      "bottom_lip": face_landmarks['bottom_lip'],
      "teeth_appearance": {
          "LAB": lab_c,
          "LUV": luv_c
      }
  })
  end_time = time.time()
  print("\tcompute_features {}: {}".format(frame_number,(end_time-start_time)))

# extract_lips
def extract_features(ifn,ofn,write_output_movie=False):
  print("Processing:",ifn)
  # ofn = ifn+"-output.avi" # It only works with AVI
  
  input_movie = cv2.VideoCapture(ifn)
  if not input_movie.isOpened(): 
      print("could not open :",ifn)

  length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_width  = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps    = input_movie.get(cv2.CAP_PROP_FPS)
  codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))

  print(input_movie)
  print("CODEC:",codec)
  print("FPS:",fps)
  print("Dimension:",frame_width,frame_height)
  print("Length:",length)
  
  step_marker = int(length / 10)

  fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
  # output_movie = cv2.VideoWriter(ofn, codec, fps, (frame_width, frame_height))

  if write_output_movie:
    output_movie = cv2.VideoWriter(ofn, fourcc, fps, (frame_width, frame_height))
    print("Output:",output_movie)
  # output_movie.release()

  # Initialize variables
  # face_locations = []
  frame_number = 0

  lip_features = []
  frame = None
  observe_frame = 100
  total_time = 0
  processes = []
  start_time = time.time()    
  while True:
    
    frame_number += 1
    
    ret, frame = input_movie.read() # Grab a single frame of video
    # end_time = time.time()
    # print("\tLoad frame {}: {}".format(frame_number,(end_time - start_time)))

    # Quit when the input video file ends
    if not ret:
      break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    # face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

    p = Process(target=compute_features, args=(frame_number,frame,lip_features,))
    processes.append(p)
    p.start()


    
#################################
#     face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

#     # Save lip features locations
#     # Assume only single face
#     face_landmarks = face_landmarks_list[0]
    

#     mouthImage,lip_landmarks = getMouthImage(rgb_frame)
#     score = getTeethScore(mouthImage,lip_landmarks)
# #     print('LAB {}\nLUV {}'.format(score[3],score[4]))
    
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
  #     print("Writing frame {} / {}".format(frame_number, length))
    print("#",end='')
    if frame_number % step_marker == 0:
      print(" %d/%d"%(frame_number,length))

    if write_output_movie:   
    #     i/len(some_list)*100," percent complete         \r",      
      # Drawing mouth image on top of the face
      # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
      x_offset = y_offset = float('inf')
      for x,y in face_landmarks_list[0]['top_lip']:
        x_offset = min(x_offset,x)
        y_offset = min(y_offset,y)
      
      markedMouthImage = markedMouthImage[:, :, ::-1]
      frame[y_offset:y_offset+markedMouthImage.shape[0], x_offset:x_offset+markedMouthImage.shape[1]] = markedMouthImage
  #     if frame_number == observe_frame: break     
      output_movie.write(frame)
#     output_movie.write(markedMouthImage)
  end_time = time.time()

  for p in processes:
    p.join()

  print("Elapse Time: {}".format(end_time - start_time))

  if write_output_movie:   
    output_movie.release()
#   plt.imshow(frame[:, :, ::-1])

  import json as j
  # outputFilename = ifn+".json"
  # with open(outputFilename,"w") as f:
  with open(ofn,"w") as f:
    j.dump(lip_features,f)
  
  # return outputFilename
  
   


def vid2vec(v):
  """Main entry for vid2vec function. will generate a JSON file of property vector of the given video

  Args:
    v (str): string

  Returns:
    int: -1 video does not exist
  """  
  # clips = ['v1.mp4','v2.mp4','v3.mp4']
  # clips = ['v5.mp4'] # The best word separation with pauses. But teeth too dark
  # clips = [v]
  # for c in clips:
  #   extract_lips(c)  
    
  try:
    with open(v) as f:
      current_dir = os.getcwd()
      basename = os.path.basename(v)
      sep = os.path.sep
      ofn = "{}{}{}.json".format(current_dir,sep,basename)
      print("\tOutput to "+ofn)
      extract_features(v,ofn)

      return ofn
  except IOError:
    _logger.warn('File "{}" not accessible'.format(v))
    return -1

  return v

def parse_args(args):
  """Parse command line parameters

  Args:
    args ([str]): command line parameters as list of strings

  Returns:
    :obj:`argparse.Namespace`: command line parameters namespace
  """
  parser = argparse.ArgumentParser(
      description="Generate a vector file of a video clip.")
  parser.add_argument(
      "--version",
      action="version",
      version="vid2vec {ver}".format(ver=__version__))
  parser.add_argument(
      dest="v",
      help="video filename",
      type=str,
      metavar="FILENAME")        
  # parser.add_argument(
  #     dest="n",
  #     help="n-th Fibonacci number",
  #     type=int,
  #     metavar="INT")
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
  """Main entry point allowing external calls

  Args:
    args ([str]): command line parameter list
  """
  args = parse_args(args)
  setup_logging(args.loglevel)
  _logger.debug("Starting crazy calculations...")
  # print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
  ret = vid2vec(args.v)
  print("RET {}".format(ret))
  _logger.info("Script ends here")


def run():
  """Entry point for console_scripts
  """
  main(sys.argv[1:])


if __name__ == "__main__":
  run()
