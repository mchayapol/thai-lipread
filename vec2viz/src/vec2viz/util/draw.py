import numpy as np
import cv2


def plot(data, frame_number=0):

    margin = 100
    radius = 2
    thickness = 2
    color0 = (0,0,0)
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)  # BGR
    background_color = 200

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 0.5

    frame = data[frame_number]

    # Fixing random state for reproducibility

    print('\nframe\n=====\n', frame)
    top_lip = np.asarray(frame['top_lip'])
    print('\ntop_lip\n=======\n', top_lip)
    bottom_lip = np.asarray(frame['bottom_lip'])

    X = top_lip[:, 0]
    Y = bottom_lip[:, 1]
    w = X.max()
    h = Y.max()
    print("X", X)
    print("Y", Y)
    print("Dimension {}x{}".format(w, h))

    # translate to origin

    x0 = X.min()
    y0 = Y.min()

    X = X - x0 + margin
    Y = Y - y0 + margin
    w = X.max() + margin
    h = Y.max() + margin

    # w = top_lip[:, 0].max()
    # h = top_lip[:, 1].max()
    print("New Dimension {}x{}".format(w, h))

    # translate = np.vectorize(lambda p: [p[0]-x0+margin,p[1]-y0+margin])
    # top_lip = translate(top_lip)
    print("New top_lip:\n============\n", top_lip)
    # print("Dimension: {}x{}".format(X.max(),Y.max()))

    img = np.zeros([h, w, 3], dtype=np.uint8)
    # img.fill(200)
    img[:] = background_color
    print("X:",X)

    i = 0
    for p in top_lip:
        print(p)
        p = (p[0]-x0+margin,p[1]-y0+margin)
        img = cv2.circle(img, p, radius, color1, thickness)
        p = (p[0]+5,p[1]-15)
        img = cv2.putText(img, str(i), p, font,  fontScale, color0, 1, cv2.LINE_AA) 
        i+=1

    i = 0
    for p in bottom_lip:
        print(p)
        p = (p[0]-x0+margin,p[1]-y0+margin)
        img = cv2.circle(img, p, radius, color1, thickness)
        p = (p[0]+5,p[1]+15)
        img = cv2.putText(img, str(i), p, font,  fontScale, color0, 1, cv2.LINE_AA) 
        i+=1

    # img = cv2.circle(img, (400,800), 20, (0, 0, 0), 5)
    cv2.imshow("Frame {}".format(frame_number), img)
    cv2.waitKey(0)
