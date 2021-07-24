"""
Test profile box from CSV
Goal:
- Generate PB CSV from CSV (currently done in vid2vec)
"""
from mlr import util, profile_box
import pandas as pd
import numpy as np
filename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v02.csv"
output_csv_filename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v02.pb2.csv"
df = pd.read_csv(filename)

original_landmark_list = util.csv2pred(filename)
print(original_landmark_list.shape)
lip_features = []
for index,row in df.iterrows():
  points = util.df_row_to_pred(row)
  landmarks = np.array(points)
  # print(landmarks)
  corrected_landmarks = profile_box.fix_profile_box(landmarks)
  # print(row)
  fl2_pb = util.prepare_df_row(corrected_landmarks, row['frame#'], row['teeth_LAB'], row['teeth_LUV'])
  lip_features.append(fl2_pb)

util.export_to_csv(lip_features,output_csv_filename)