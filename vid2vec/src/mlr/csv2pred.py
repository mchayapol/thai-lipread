
from re import X


def csv2pred(filename):
  """
  return numpy array matching 3D
  """
  import pandas as pd
  import numpy as np
  frames = []
  
  print(filename)
  df = pd.read_csv(filename)
  print(df.columns)
  
  for i in range(1,69):
    x = f"{i}_x"
    y = f"{i}_y"
    z = f"{i}_z"
    # df = df[[x,y,z]]
    coord = df[[x,y,z]].iloc[i].values.flatten().tolist()
    # coord = df[[x,z,y]].iloc[i].values.flatten().tolist()
    # print(coord)
    frames.append(coord)

  # print(np.array(frames))
  return np.array(frames)
  
    

def __what_is_in_pred(image_filename):
  import face_alignment
  from skimage import io
  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')
  input = io.imread(image_filename)
  preds = fa.get_landmarks(input)
  detection = preds[0]
  # print(type(detection))
  # print(len(detection))
  print(detection)

if __name__ == "__main__":
  __what_is_in_pred("face1.jpg")

  filename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/chayapol/v1.csv"
  detection = csv2pred(filename)

  # __what_is_in_pred(image_filename):