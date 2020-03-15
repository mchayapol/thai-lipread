import numpy as np
import cv2
from . import draw


def stabilize(data):
    """
    Use the left and right most edge of lips to find the center
    """
    datas = fix_slope(data)

    #   c_center, datac = find_center(data)
    #   print("Clip center: {}".format(c_center))
    # print(datac[0]['center'])

    #   datas = shift_center(datac,c_center)

    return datas


def fix_slope(data):
    """
    Parameters
    ==========
    data is a list of frame with property "center"

    Return
    ======
    datas : shifted frames
    """
    datas = []
    max_slope = 0
    max_slope_frame = 0
    for f in data:
        top_lip = f['top_lip']
        bottom_lip = f['bottom_lip']
        # Find slope
        pT0 = top_lip[0]
        pB0 = bottom_lip[0]
        (x0, y0) = pT0
        (x1, y1) = pB0
        dy, dx = (y1-y0), (x1-x0)
        slope = dy/dx
        if max_slope < slope:
            max_slope = slope
            max_slope_frame = f['frame_id']
            dyN,dxN = dy,dx
        print("Slope #{} {}/{} = {}".format(f['frame_id'], pB0, pT0, slope))

        datas.append(f)

    print("datas.size: ", len(datas))
    print("Slope #{} = {}".format(max_slope_frame, max_slope))
    draw.plot(data, frame_id=max_slope_frame)

    angle = np.rad2deg(np.arctan2(dyN, dxN))
    draw.plot(data, frame_id=max_slope_frame, angle=angle)
    # TODO rotate the cordinates in list

    draw.plot(datas, frame_id=max_slope_frame)
    return datas
