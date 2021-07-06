# import the opencv library
import cv2
import face_alignment
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def drawLines(image,points,color=(200,255,200),thickness=2):
  for i in range(1,len(points)):
    p1 = tuple(points[i-1][0:2])
    p2 = tuple(points[i][0:2])
    print(p1,p2)
    image = cv2.line(image,p1,p2,color,thickness)
  return image

def drawPoints(image,points):
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
  image = drawLines(image,points[0:17])
  print(points)
  pts = np.array(points, np.int32)
  pts = np.delete(pts,2,1)
  print(pts)
  image2 = cv2.polylines(image,[pts],True,(200,200,255))  
  return image2
      
  
# define a video capture object
# vid = cv2.VideoCapture(0)
  
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False,device='cpu')

# input  = io.imread('image01.jpg')
input = mpimg.imread('image01.jpg')
preds = fa.get_landmarks(input)
frame2 = drawPoints(input,preds[0])
cv2.imshow('frame', frame2)
# plt.imshow(frame2)
# while(True): pass

# while(False):
      
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()



#     preds = fa.get_landmarks(frame)

#     frame2 = drawPoints(frame,preds[0])

#     # Display the resulting frame
#     cv2.imshow('frame', frame2)
      
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
  
# vid.release()
# cv2.destroyAllWindows()