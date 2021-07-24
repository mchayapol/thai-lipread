import matplotlib.pyplot as plt
from mlr import util
import pandas as pd
import sys

csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.pb.csv"
csvFilename = sys.argv[1]

util.draw2D_from_CSV(csvFilename)
