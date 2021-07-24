
def __find_angle(p1, p2):
    # print(f"Angle between {p1} {p2}")
    unit_vector_1 = p1 / np.linalg.norm(p1)
    unit_vector_2 = p2 / np.linalg.norm(p2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radian = np.arccos(dot_product)
    angle = math.degrees(radian)
    return angle


def doPlot(ax, data):
    # ax.scatter(data[:,1], data[:,0], data[:,2],zdir='z')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z',c="black")
    ax.view_init(90, 90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot3D(data, data2=None, data3=None, data4=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 14))
    # fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    doPlot(ax, data)

    if data2 is not None:
        ax = fig.add_subplot(222, projection='3d')
        doPlot(ax, data2)

    if data3 is not None:
        ax = fig.add_subplot(223, projection='3d')
        doPlot(ax, data3)

    if data4 is not None:
        ax = fig.add_subplot(224, projection='3d')
        doPlot(ax, data3)

    plt.show()


def find_angle(p1, p2):
    """
    return angle in degree
    """
    import math
    # print(f"Angle between {p1} {p2}")
    v = (x, y) = p2-p1
    # print("\tVector", v)
    radian = math.atan2(y, x) - math.pi
    degree = math.degrees(radian)
    return degree

def fix_profile_box(landmarks):
    """
    landmarks: numpy array
    Only 2 rotation is enough!
    """
    # print(landmarks)
    import math
    from scipy.spatial.transform import Rotation as R
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    anchor_points = [48, 54]  # mouth corners
    vector_1, vector_2 = landmarks[anchor_points]
    angleX = angleY = angleZ = 0
    corrected_landmarks = landmarks


    # y-axis, turn left-right
    p1,p2 = vector_1[[0, 2]],vector_2[[0, 2]] # ORG
    angleY = find_angle(p1, p2)+180

    r = R.from_euler('xyz', (0, angleY, 0), degrees=True)
    corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points

    # x-axis, head up down
    vector_1, vector_2 = corrected_landmarks[anchor_points]
    p1,p2 = vector_1[[1, 2]],vector_2[[1, 2]]
    p1,p2 = vector_1[[2, 1]],vector_2[[2, 1]]
    angleX = find_angle(p1, p2)
    r = R.from_euler('xyz', (angleX, 0, 0), degrees=True)
    # corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points

    # z-axis, head side-to-side
    vector_1, vector_2 = corrected_landmarks[anchor_points]
    p1,p2 = vector_1[[0, 1]],vector_2[[0, 1]] # ORG
    p1,p2 = vector_1[[1, 0]],vector_2[[1, 0]]
    angleZ = find_angle(p1, p2) + 90
    r = R.from_euler('xyz', (0, 0, angleZ), degrees=True)
    corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points

    # print(f"{vector_1}\n{vector_2}")
    # print(f"Angle Y: {angleY}")
    # print(f"Angle X: {angleX}")
    # print(f"Angle Z: {angleZ}")

    return corrected_landmarks

def fix_profile_box_ORG(landmarks):
    """
    Only 2 rotation is enough!
    """
    # print(landmarks)
    import math
    from scipy.spatial.transform import Rotation as R
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # anchor_points = [0, 16]  # face corners
    anchor_points = [48, 54]  # mouth corners
    # anchor_points = [54, 48]  # mouth corners
    # print(f"Anchor Points: {anchor_points}")
    vector_1, vector_2 = landmarks[anchor_points]
    print(f"{vector_1}\n{vector_2}")

    angleX = angleY = angleZ = 0
    corrected_landmarks = landmarks
    # y-axis, turn left-right
    p1,p2 = vector_1[[0, 2]],vector_2[[0, 2]] # ORG

    # angleY = find_angle(p1,p2) + 360
    # angleY = find_angle(p1, p2) - 180  # org
    angleY = find_angle(p1, p2)
    angleY = 0
    # angleY = abs(angleY)
    # if 90 < abs(angleY) < 180:
    #     angleY = abs(angleY) - 180

    print(f"Angle Y: {angleY}")
    r = R.from_euler('xyz', (0, angleY, 0), degrees=True)
    corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points

    # x-axis, head up down
    vector_1, vector_2 = corrected_landmarks[anchor_points]
    p1 = vector_1[[1, 2]]
    p2 = vector_2[[1, 2]]

    p1 = vector_1[[2, 1]]
    p2 = vector_2[[2, 1]]
    # angleX = find_angle(p1, p2) + 360 # org
    angleX = find_angle(p1, p2)
    # if int(abs(angleX)) == 90:
    #     angleX = 0
    # elif abs(angleX) > 90:
    #     angleX -= 180

    # angleX = 180
    print(f"Angle X: {angleX}")
    r = R.from_euler('xyz', (angleX, 0, 0), degrees=True)
    # corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points

    # z-axis, head side-to-side
    vector_1, vector_2 = corrected_landmarks[anchor_points]
    p1,p2 = vector_1[[0, 1]],vector_2[[0, 1]] # ORG
    p1,p2 = vector_1[[1, 0]],vector_2[[1, 0]]
    # angleZ = find_angle(p1, p2) * -1
    # angleZ = find_angle(p1, p2) * -1 + 180  # ORG
    angleZ = find_angle(p1, p2) + 90
    # angleZ = abs(angleZ)
    # angleZ = -90
    print(f"Angle Z: {angleZ}")
    r = R.from_euler('xyz', (0, 0, angleZ), degrees=True)
    corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points

    # print(f"Angle Y: {angleY}")
    # print(f"Angle X: {angleX}")
    # print(f"Angle Z: {angleZ}")

    # r = R.from_euler('xyz',(angleX,angleY,angleZ), degrees=True)
    # corrected_r = r.apply(detection) #Rotated points
    # print("Before\n", detection[anchor_points])
    # print("After\n", corrected_r[anchor_points])


    return corrected_landmarks

def fix_profile_box_1(landmarks):
    """
    doesn't work
    """
    # print(landmarks)
    import math
    from scipy.spatial.transform import Rotation as R
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    anchor_points = [48, 54]  # mouth corners
    vector_1, vector_2 = landmarks[anchor_points]
    print(f"{vector_1}\n{vector_2}")
    angleX = angleY = angleZ = 0
    corrected_landmarks = landmarks

    p1,p2 = vector_1[[1, 2]],vector_2[[1, 2]]
    angleX = find_angle(p1, p2)

    p1,p2 = vector_1[[0, 2]],vector_2[[0, 2]]
    angleY = find_angle(p1, p2)

    p1,p2 = vector_1[[0, 1]],vector_2[[0, 1]]
    angleZ = find_angle(p1, p2)

    print(f"Angles: {(angleX, angleY, angleZ)}")
    r = R.from_euler('xyz', (angleX, angleY, angleZ), degrees=True)
    corrected_landmarks = r.apply(corrected_landmarks)  # Rotated points



    return corrected_landmarks

def test_profile_box_single_frame(filename,frame_no):
    """
    df Pandas DataFrame 3D
    return Pandas DataFrame
    """
    from util import csv2pred
    original_landmark_frames = csv2pred(filename)
    print(original_landmark_frames.shape)

    corrected_landmarks = fix_profile_box(original_landmark_frames[frame_no])
    print(corrected_landmarks.shape)
    return original_landmark_frames, corrected_landmarks
    


if __name__ == "__main__":
    import face_alignment


    frame_no = 700 # Problem frame
    frame_no = 100
    frame_no = 200
    frame_no = 600
    filename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.csv"
    print(f"Frame No: {frame_no}")

    original_landmark_frames, corrected_landmarks = test_profile_box_single_frame(filename,frame_no)
    # print(f"Frames: {len(original_landmark_frames)}")
    # print(original_landmarks[0])
    # print(corrected_landmarks[frame_no])
    plot3D(original_landmark_frames[frame_no], corrected_landmarks)
