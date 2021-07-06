"""
Does not work with "CUDA"
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
"""
import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

input = io.imread('image01.jpg')
preds = fa.get_landmarks(input)