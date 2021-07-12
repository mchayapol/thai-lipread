# -*- coding: utf-8 -*-
"""
Display visualise of points from CSV
"""

import argparse
import sys
import logging
# from .util import method1, method2, draw, vec2vid
import cv2
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vec2viz import __version__

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class VFace:
  def __init__(self,data,width=800,height=800):
    self.width = width
    self.height = height
    self.teeth_LAB = data['teeth_LAB']
    self.teeth_LUV = data['teeth_LUV']
    self.points = [0] * 68
    self.minx = float('inf')
    self.miny = float('inf')
    self.minz = float('inf')
    self.maxx = float('-inf')
    self.maxy = float('-inf')
    self.maxz = float('-inf')
    for i in range(1,69):
      x = data[f"{i}_x"]
      y = data[f"{i}_y"]
      z = data[f"{i}_z"]
      self.minx = min(self.minx,x)
      self.miny = min(self.miny,y)
      self.minz = min(self.minz,z)
      self.maxx = max(self.maxx,x)
      self.maxy = max(self.maxy,y)
      self.maxz = max(self.maxz,z)
      
      self.points[i-1] = (x,y,z)
      # print(x,y,z)
      # print(i)

  def getImage(self):
    image = np.zeros((self.height,self.width,3), np.uint8)
    for i in range(0,68):
      x,y,z = self.points[i]
      # print(i,x,y)
      image = cv2.circle(image,(int(x),int(y)),2,(255,255,255),2)
    return image


def visualise(vector_file):
  """
  Display frame-by-frame from vector file
  to validate the correctness of feature extraction.
  This can also use to view after Profile Box.
  """
  _logger.info("Input file: {}".format(vector_file))
  df = pd.read_csv(vector_file)  
  cols = df.columns
  print(cols)
  for index, row in df.iterrows():
    # print(row['frame#'], row['1_x'])
    print(row)
    vf = VFace(row)
    img = vf.getImage()
    cv2.imshow('Face', img)
    # plt.imshow(image)
    # plt.show()
    
  # with open(vector_file) as csvfile:
  #   reader = csv.reader(csvfile, delimiter=',')
  #   for row in reader:
  #     print(row)
  #     exit()


def parse_args(args):
  """Parse command line parameters

  Args:
    args ([str]): command line parameters as list of strings

  Returns:
    :obj:`argparse.Namespace`: command line parameters namespace
  """
  parser = argparse.ArgumentParser(
    description="Convert a vector file to a viseme")
  parser.add_argument(
    "--version",
    action="version",
    version="vec2viz {ver}".format(ver=__version__))
  parser.add_argument(
    dest="vector_file",
    help="vector file (JSON)",
    type=str,
    metavar="FILENAME")
  parser.add_argument(
    "-v",
    "--verbose",
    dest="loglevel",
    help="set loglevel to INFO",
    action="store_const",
    const=logging.INFO)
  parser.add_argument(
    "-vv",
    "--very-verbose",
    dest="loglevel",
    help="set loglevel to DEBUG",
    action="store_const",
    const=logging.DEBUG)
  return parser.parse_args(args)


def setup_logging(loglevel):
  """Setup basic logging

  Args:
    loglevel (int): minimum loglevel for emitting messages
  """
  logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
  logging.basicConfig(level=loglevel, stream=sys.stdout,
            format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
  """Main entry point allowing external calls

  Args:
    args ([str]): command line parameter list
  """
  args = parse_args(args)
  setup_logging(args.loglevel)
  _logger.debug("Starting crazy calculations...")
  # print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
  visualise(args.vector_file)
  _logger.info("Script ends here")


def run():
  """Entry point for console_scripts
  """
  main(sys.argv[1:])


if __name__ == "__main__":
  run()
