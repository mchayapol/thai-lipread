import cv2
import face_alignment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from . import util
from shapely.geometry import Polygon

class Face:
    def __init__(self, image, points):
        self.image = image
        self.parts = {
            "chin": points[0:17],
            "eyebrow_left": points[17:22],
            "eyebrow_right": points[22:27],
            "nose": points[27:31],
            "nostril": points[31:36],
            "eye_left": points[36:42],
            "eye_right": points[42:48],
            "lip_outter": points[48:60],
            "lip_inner": points[60:68]
        }

    def getPoints(self, part):
        try:
            return self.parts[part]
        except KeyError:
            raise Exception(f"part must be either {self.parts.keys()}.")

    def showFace(self, figsize=(8, 6)):
        image = self.image
        image = util.drawPoints(image, preds[0])
        figure(figsize=figsize, dpi=80)
        plt.imshow(image)
        plt.show()

    def getMouthImage(self, margin=0):
        minx = miny = float('inf')
        maxx = maxy = float('-inf')

        lip_otter_points = self.parts['lip_outter']
        for x, y, z in lip_otter_points:
            # print(x,y)
            minx = int(min(minx, x))
            miny = int(min(miny, y))
            maxx = int(max(maxx, x))
            maxy = int(max(maxy, y))

        # print("------------- ",(minx,miny),(maxx,maxy))
        # print("------------- ",(miny-margin,maxy+margin),(minx-margin,maxx+margin))
        mouthImage = self.image[miny-margin:maxy +
                                margin, minx-margin:maxx+margin]

        # Translate landmarks relative to margin x,y
        lip_bounds = []
        mouth_cavity_landmarks = []

        for p in self.parts['lip_outter']:
            p2 = (p[0] - minx, p[1] - miny)
            lip_bounds.append(p2)

        for p in self.parts['lip_inner']:
            p2 = (p[0] - minx, p[1] - miny)
            mouth_cavity_landmarks.append(p2)
        
        # print("translated_lip_landmarks",translated_lip_landmarks)
        return mouthImage, lip_bounds, mouth_cavity_landmarks

    def getTeethScore(self):
        mouthImage, lip_bounds,mouth_cavity_bounds = self.getMouthImage()
        height, width, channels = mouthImage.shape
        print(f"\tMouth Image Shape {width, height}")
        area_mouth = height * width # WRONG this has to be polygon area of the mouth only, not rectangular
        # print(lip_bounds)
        # print(lip_bounds[0])
        # print(lip_bounds[1])
        pgon = Polygon(lip_bounds)
        print("pgon.area",pgon.area)


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
                if self.isin_inner_mouth(mouth_cavity_bounds, x, y):
                    # print("IN MOUTH")
                    p = row[x]
                    lab_a = lab2[y, x, 1]
                    luv_a = luv2[y, x, 1]
                    if lab_a <= ta: # Pink
                        p[0] = 200  # L
                        # p[1] = 255  # L
                        p[2] = 200  # L
                        lab_c += 1
                    if luv_a <= tu: # Yellow
                        p[0] = 200  # L
                        p[1] = 200  # L
                        # p[2] = 200  # L
                        luv_c += 1

        tr_lab = lab_c/pgon.area
        tr_luv = luv_c/pgon.area
        # print(lab_c, luv_c, area_mouth, pgon.area)
        # print(f"Ratio 1:\n\tLAB_C {lab_c/area_mouth}\n\tLUV_C {luv_c/area_mouth}")
        # print(f"Ratio 2:\n\tLAB_C {tr_lab}\n\tLUV_C {tr_luv}")
        # Teeth Ratio
        return (hilightedMouthImage, lab, luv, lab_c, luv_c, tr_lab, tr_luv)

    def isin_inner_mouth(self, lip_boundary, x, y):
        """
        isin_inner_mouth (from TLR Teeth Appearance Calculation.ipynb)
        """
        bounds = lip_boundary
        isin = util.ray_tracing_method(x, y, bounds)
        return isin

        # face_recognition
        top_lip = lip_boundary['top_lip']
        bottom_lip = lip_boundary['bottom_lip']
        bounds = np.concatenate((top_lip[6:], bottom_lip[6:]), axis=0)
        isin = ray_tracing_method(x, y, bounds)
        return isin


if __name__ == '__main__':
    import matplotlib.image as mpimg
    import face_alignment
    import time
    s = time.time()
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda')
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda:0')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=False, device='cpu')

    image = mpimg.imread('../../face01.jpg')
    preds = fa.get_landmarks(image)
    e = time.time()
    print(f"fa.get_landmarks processing time: {e-s}")
    f = Face(image, preds[0])
    # f.showFace()

    # print(f.getPoints('lip_outter'))
    # image = f.getMouthImage()

    (image, lab, luv, lab_c, luv_c) = f.getTeethScore()
    print(f"LAB_C {lab_c}\nLUV_C {luv_c}")
    plt.imshow(image)
    plt.show()
