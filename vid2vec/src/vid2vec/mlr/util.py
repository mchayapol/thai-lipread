import cv2
import face_alignment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure


def ray_tracing_method(x, y, poly):
    """
    Ray Tracing method
    """
    n = len(poly)
    # print(f"ray_tracing_method n = {n}")
    inside = False   
    p1x, p1y = poly[0]
    # print(x,y,p1x,p1y)
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def drawLines(image, points, color=(200, 255, 200), thickness=3):
    for i in range(1, len(points)):
        p1 = tuple(points[i-1][0:2])
        p2 = tuple(points[i][0:2])
        # print(p1,p2)
        image = cv2.line(image, p1, p2, color, thickness)
    return image


def drawPoints(image, points):
    image = image[:]
    locations = {
        "chin": points[0:17],
        "eyebrow_left": points[17:22],
        "eyebrow_right": points[22:27],
        "eye_left": np.vstack([points[36:42], points[36]]),
        "eye_right": np.vstack([points[42:48], points[42]]),
        "nose": points[27:31],
        "nose_under": points[31:36],
        "lip_outter": np.vstack([points[48:60], points[48]]),
        "lip_outter": np.vstack([points[60:68], points[60]]),
    }
    for key in locations:
        # print(key)
        L = locations[key]
        image = drawLines(image, L)
    return image


if __name__ == '__main__':
    frame = input[:]
    frame2 = drawPoints(frame, preds[0])
    figure(figsize=(8, 6), dpi=80)
    plt.imshow(frame2)
    plt.show()
