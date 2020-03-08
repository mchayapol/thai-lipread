import face_recognition
import cv2
import numpy as np

# getMouthImage (from TLR Teeth Appearance Calculation.ipynb)
def getMouthImage(faceImage,margin=0):
  # face_locations = face_recognition.face_locations(faceImage)
  face_landmarks_list = face_recognition.face_landmarks(faceImage)
  if len(face_landmarks_list) == 0:
      return None

  
  minx = miny = float('inf')
  maxx = maxy = float('-inf')

  for x,y in face_landmarks_list[0]['top_lip']:
    minx = min(minx,x)
    miny = min(miny,y)

  for x,y in face_landmarks_list[0]['bottom_lip']:
    maxx = max(maxx,x)
    maxy = max(maxy,y)

  mouthImage = faceImage[miny-margin:maxy+margin,minx-margin:maxx+margin]
  
  # lip_landmarks must be translate to origin (0,0) by minx, miny  

  lip_landmarks = {
      'top_lip': [],
      'bottom_lip': []
  }

  for p in face_landmarks_list[0]['top_lip']:
    p2 = (p[0] - minx, p[1] - miny)
    lip_landmarks['top_lip'].append(p2)

  for p in face_landmarks_list[0]['bottom_lip']:
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


def extract_features(image):
    frame = image
    rgb_frame = frame[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    if len(face_landmarks_list) == 0:
        return None
    face_landmarks = face_landmarks_list[0]
    

    mouthImage,lip_landmarks = getMouthImage(rgb_frame)
    score = getTeethScore(mouthImage,lip_landmarks)
    
    markedMouthImage = score[0]
    lab_c = score[3]
    luv_c = score[4]
    
    lip_features = {
        # "frame_id": frame_number,
        "top_lip": face_landmarks_list[0]['top_lip'],
        "bottom_lip": face_landmarks_list[0]['bottom_lip'],
        "teeth_appearance": {
            "LAB": lab_c,
            "LUV": luv_c
        }
    }

    x_offset = y_offset = float('inf')
    for x,y in face_landmarks_list[0]['top_lip']:
      x_offset = min(x_offset,x)
      y_offset = min(y_offset,y)
    
    markedMouthImage = markedMouthImage[:, :, ::-1]
    frame[y_offset:y_offset+markedMouthImage.shape[0], x_offset:x_offset+markedMouthImage.shape[1]] = markedMouthImage
    return frame,lip_features

