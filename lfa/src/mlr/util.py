import logging
_logger = logging.getLogger(__name__)

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


def csv2pred(filename):
    """
    return numpy array matching 3D
    """
    import pandas as pd
    # import numpy as np
    # frames = []

    print(filename)
    df = pd.read_csv(filename)
    # print(df.columns)
    return df2pred(df)

    # for i in range(1,69):
    #   x = f"{i}_x"
    #   y = f"{i}_y"
    #   z = f"{i}_z"
    #   # df = df[[x,y,z]]
    #   coord = df[[x,y,z]].iloc[i].values.flatten().tolist()
    #   # coord = df[[x,z,y]].iloc[i].values.flatten().tolist()
    #   # print(coord)
    #   frames.append(coord)

    # # print(np.array(frames))
    # return np.array(frames)

def export_to_csv(lip_features,output_csv_filename):
    import csv
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
    _logger.info(f".... Wrote {row_no} records to {output_csv_filename}")
    print(f".... Wrote {row_no} records to {output_csv_filename}")


def prepare_df_row(landmarks, frame_number, tr_lab, tr_luv):
    prepared_landmarks = {}
    prepared_landmarks["frame#"] = frame_number
    point_no = 1
    for point in landmarks:
        x = f"{point_no}_x"
        y = f"{point_no}_y"
        z = f"{point_no}_z"

        prepared_landmarks[x] = point[0]
        prepared_landmarks[y] = point[1]
        prepared_landmarks[z] = point[2]
        point_no += 1

    prepared_landmarks["teeth_LAB"] = tr_lab
    prepared_landmarks["teeth_LUV"] = tr_luv
    return prepared_landmarks

def df_row_to_pred(row):
    points = []
    for pno in range(1, 69):
        x = f"{pno}_x"
        y = f"{pno}_y"
        z = f"{pno}_z"
        points.append([row[x],row[y],row[z]])
        # df = df[[x,y,z]]
        # coord = df[[x, y, z]].iloc[i].values.flatten().tolist()
        # coord = df[[x,z,y]].iloc[i].values.flatten().tolist()
        # print(coord)
    return points

def df2pred(df):
    """
    return numpy array matching 3D
    """

    import numpy as np
    frames = []
    for index,row in df.iterrows():
        points = df_row_to_pred(row)
        frames.append(points)

    # print(np.array(frames))
    return np.array(frames)


def draw2D_from_CSV(csvFilename,delay=.1):
    import pandas as pd
    df = pd.read_csv(csvFilename)
    draw2D_from_df(df)

def draw2D_from_df(df,delay=.05):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    plt.ion()
    fig, ax = plt.subplots()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for index, row in df.iterrows():
        frameno = row['frame#']
        textstr = f"Frame#: {int(frameno)}"
        X,Y = [],[]
        for pno in range(1, 69):
            x = f'{pno}_x'
            y = f'{pno}_y'
            # print((row[x],row[y]))
            X.append(row[x])
            Y.append(1000-row[y])   # flip the plot

        sc = ax.scatter(X, Y, color='black')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        # print(len(X),len(Y))

        # sc.set_offsets(np.c_[X,Y])
        plt.xlim([0,1000])
        plt.ylim([0,1000])

        fig.canvas.draw_idle()
        plt.pause(delay)
        ax.cla()

    plt.waitforbuttonpress()



if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.image as mpimg
    from matplotlib.pyplot import figure

    # frame = input[:]
    # frame2 = drawPoints(frame, preds[0])
    # figure(figsize=(8, 6), dpi=80)
    # plt.imshow(frame2)
    # plt.show()

    # csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.pb.csv"
    # csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.csv"
    csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v02.pb.csv"
    filename, file_extension = os.path.splitext(csvFilename)
    print(f"{filename}.{file_extension}")
    draw2D_from_CSV(csvFilename)
