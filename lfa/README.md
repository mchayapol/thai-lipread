# Lip Feature Analysis (LFA)


# Description
A longer description of your project goes here...


# Note
This project has been set up using PyScaffold 3.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.


# Setup
python setup.py develop

# Sample
```
lfa -m 4 --label v01 D:\GoogleDrive-VMS\Research\lip-reading\datasets\angsawee\avi\run-2021-01-17\v01.csv
```
Quiet mode
```
lfa -q -m 4 --label v01 D:\GoogleDrive-VMS\Research\lip-reading\datasets\angsawee\avi\run-2021-01-17\v01.csv
```


# MLR Lib
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
## Visualise
```
visualise "D:\GoogleDrive-VMS\Research\lip-reading\datasets\chayapol\v1.csv"
```
