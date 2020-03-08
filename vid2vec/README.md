
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
python setup.py develop
vid2vec "D:\GoogleDrive-SCITECH\Research\lip-read\datasets\clips\v1.mp4"