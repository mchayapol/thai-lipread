import numpy as np
import cv2

from tlr import teeth

cap = cv2.VideoCapture(0)
cv2.startWindowThread()

while(True):
    ret, frame = cap.read()
    
    ret = teeth.extract_features(frame)

    # cv2.equalizeHist(frame)        
    # Display the resulting frame
    if ret is not None:        
        frame,lip_features = ret
        cv2.imshow('Processed',frame)
    else:
        cv2.imshow('Processed',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()