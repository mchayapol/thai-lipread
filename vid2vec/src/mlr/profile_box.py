
def __find_angle(p1, p2):
    # print(f"Angle between {p1} {p2}")
    unit_vector_1 = p1 / np.linalg.norm(p1)
    unit_vector_2 = p2 / np.linalg.norm(p2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radian = np.arccos(dot_product)
    angle = math.degrees(radian)
    return angle


def __doPlot(ax, data):
    # ax.scatter(data[:,1], data[:,0], data[:,2],zdir='z')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z')
    ax.view_init(90, 90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot3D(data, data2=None, data3=None, data4=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 14))
    # fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    __doPlot(ax, data)

    if data2 is not None:
        ax = fig.add_subplot(222, projection='3d')
        __doPlot(ax, data2)

    if data3 is not None:
        ax = fig.add_subplot(223, projection='3d')
        __doPlot(ax, data3)

    if data4 is not None:
        ax = fig.add_subplot(224, projection='3d')
        __doPlot(ax, data3)

    plt.show()


def find_angle(p1, p2):
    import math
    # print(f"Angle between {p1} {p2}")
    v = (x, y) = p2-p1
    # print("\tVector", v)
    radian = math.atan2(y, x) - math.pi
    degree = math.degrees(radian)
    return degree


def fix_profile_box(detection):
    import math
    from scipy.spatial.transform import Rotation as R

    anchor_points = [0, 16]  # face corners
    anchor_points = [48, 54]  # mouth corners
    # print(f"Anchor Points: {anchor_points}")
    vector_1, vector_2 = detection[anchor_points]
    # print(vector_1, vector_2)

    angleX = angleY = angleZ = 0
    corrected_r = detection
    # y-axis
    p1 = vector_1[[0, 2]]
    p2 = vector_2[[0, 2]]
    # angleY = find_angle(p1,p2) + 360
    angleY = find_angle(p1, p2) - 180  # 172
    r = R.from_euler('xyz', (0, angleY, 0), degrees=True)
    corrected_r = r.apply(corrected_r)  # Rotated points

    # x-axis
    vector_1, vector_2 = corrected_r[anchor_points]
    p1 = vector_1[[1, 2]]
    p2 = vector_2[[1, 2]]
    angleX = find_angle(p1, p2) + 360
    if int(abs(angleX)) == 90:
        angleX = 0
    elif abs(angleX) > 90:
        angleX -= 180
    # angleX = find_angle(p1,p2) # -180
    r = R.from_euler('xyz', (angleX, 0, 0), degrees=True)
    corrected_r = r.apply(corrected_r)  # Rotated points

    # z-axis
    vector_1, vector_2 = corrected_r[anchor_points]
    p1 = vector_1[[0, 1]]
    p2 = vector_2[[0, 1]]
    # angleZ = find_angle(p1, p2) * -1
    angleZ = find_angle(p1, p2) * -1 + 180  # 172
    r = R.from_euler('xyz', (0, 0, angleZ), degrees=True)
    corrected_r = r.apply(corrected_r)  # Rotated points

    # print(f"Angle Y: {angleY}")
    # print(f"Angle X: {angleX}")
    # print(f"Angle Z: {angleZ}")

    # r = R.from_euler('xyz',(angleX,angleY,angleZ), degrees=True)
    # corrected_r = r.apply(detection) #Rotated points
    # print("Before\n", detection[anchor_points])
    # print("After\n", corrected_r[anchor_points])

    return corrected_r


def profile_box(filename):
    """
    df Pandas DataFrame 3D
    return Pandas DataFrame
    """
    detection = csv2pred(filename)
    corrected_r = fix_profile_box(detection)
    print(corrected_r)
    return detection, corrected_r
    


if __name__ == "__main__":
    import face_alignment
    from csv2pred import csv2pred
    filename = "D:\\GoogleDrive-VMS\\Research\\lip-reading\\datasets\\angsawee\\avi\\run-2021-01-17\\v01.csv"
    detection, corrected_r = profile_box(filename)
    plot3D(detection, corrected_r)
