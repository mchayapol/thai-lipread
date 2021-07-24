import numpy as np
import cv2
import logging
_logger = logging.getLogger(__name__)

# Global params
# TODO make this configurable
margin = 100
radius = 2
thickness = 2
color0 = (0,0,0)
color1 = (0, 255, 0)
color2 = (0, 0, 255)  # BGR
background_color = 200
scale = 1.0

font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 0.5

def plotf(frame,show_index=False,angle=0):
    """
    Plot a frame 
    """
    # Fixing random state for reproducibility
    _logger.debug('\nframe\n=====\n {}'.format(frame))
    top_lip = np.asarray(frame['top_lip'])
    _logger.debug('\ntop_lip\n=======\n'.format(top_lip))
    bottom_lip = np.asarray(frame['bottom_lip'])

    X = top_lip[:, 0]
    Y = bottom_lip[:, 1]
    w = X.max()
    h = Y.max()
    _logger.debug("X".format(X))
    _logger.debug("Y".format(Y))
    _logger.debug("Dimension {}x{}".format(w, h))

    # translate to origin

    x0 = X.min()
    y0 = Y.min()

    X = X - x0 + margin
    Y = Y - y0 + margin
    w = X.max() + margin
    h = Y.max() + margin

    # w = top_lip[:, 0].max()
    # h = top_lip[:, 1].max()
    _logger.debug("New Dimension {}x{}".format(w, h))

    # translate = np.vectorize(lambda p: [p[0]-x0+margin,p[1]-y0+margin])
    # top_lip = translate(top_lip)
    _logger.debug("New top_lip:\n============\n {}".format(top_lip))
    # print("Dimension: {}x{}".format(X.max(),Y.max()))

    img = np.zeros([h, w, 3], dtype=np.uint8)
    # img.fill(200)
    img[:] = background_color
    _logger.debug("X:".format(X))


    i = 0
    for p in top_lip:
        _logger.debug(p)
        p = (p[0]-x0+margin,p[1]-y0+margin)
        img = cv2.circle(img, p, radius if i != 0 else 5, color1, thickness)
        if show_index: 
            p = (p[0]+5,p[1]-15)
            img = cv2.putText(img, str(i), p, font,  fontScale, color0, 1, cv2.LINE_AA) 
        i+=1

    i = 0
    for p in bottom_lip:
        _logger.debug(p)
        p = (p[0]-x0+margin,p[1]-y0+margin)
        img = cv2.circle(img, p, radius if i != 0 else 5, color1, thickness)
        if show_index:
            p = (p[0]+5,p[1]+15)
            img = cv2.putText(img, str(i), p, font,  fontScale, color0, 1, cv2.LINE_AA) 
        i+=1

    # Rotate
    if angle != 0:
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        print(M)
        img = cv2.warpAffine(img, M, (w, h),borderMode=cv2.BORDER_REPLICATE)
    # img = cv2.circle(img, (400,800), 20, (0, 0, 0), 5)
    return img

def plot(data, frame_id=0,show_index=False,angle=0,show=True):
    _logger.debug("data.size: {}".format(len(data)))
    frame = data[frame_id]
    img = plotf(frame,show_index,angle)

    if show:
        cv2.imshow("Frame {}".format(frame_id), img)
        cv2.waitKey(0)

    return img
    