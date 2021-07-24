
from re import X
from util import csv2pred



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