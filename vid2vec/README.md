# Vid2Vec
Extract features from video

conda create -n cv python=3.6
conda activate cv

conda install -c conda-forge opencv

# Libraries
https://shakeratos.tistory.com/42
install CMake 3.4
pip install face_recognition
--- also include dlib

# Scaffold
conda install -c conda-forge pyscaffold
python setup.py develop

# Classification Model
http://scikit-learn.sourceforge.net/stable/modules/hmm.html

# Sample test calls
Prepare the shell
```
conda
activate cv
python setup.py develop
```
## for default 3D (use face_alignment)
```
vid2vec -vv "D:\GoogleDrive-VMS\Research\lip-reading\datasets\chayapol\v1.mp4"
```
## for 2D (use face_detection)
```
vid2vec -vv -2d "D:\GoogleDrive-VMS\Research\lip-reading\datasets\chayapol\v1.mp4"
```

# TODO
1. Unable to use CUDA due to this error
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```