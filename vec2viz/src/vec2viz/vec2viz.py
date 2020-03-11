# -*- coding: utf-8 -*-
"""
Convert vector to viseme
"""

import argparse
import sys
import logging

from vec2viz import __version__

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def fib(n):
  """Fibonacci example function

  Args:
    n (int): integer

  Returns:
    int: n-th Fibonacci number
  """
  assert n > 0
  a, b = 1, 1
  for i in range(n-1):
    a, b = b, a+b
  return a

def find_center(data):
  """
    Find center point of the clip
    RETURN
    ======
    a tuple of (c,d)
    c : center coordinate (x,y)
    d : data with frame's center

  """
  sumX = sumY = 0
  n = len(data)
  datac = []
  for f in data:
    f = f.copy()
    points = f["top_lip"] + f["bottom_lip"]
    x1 = x2 = points[0][0]
    y1 = y2 = points[0][1]
    for p in points:
      x,y = p
      x1 = min(x1,x)
      x2 = max(x2,x)
      y1 = min(y1,y)
      y2 = max(y2,y)
    f_center = ((x1+x2)/2, (y1+y2)/2)
    f["center"] = f_center
    datac.append(f) 
    sumX += f_center[0]
    sumY += f_center[1]

  c_center = (int(sumX/n),int(sumY/n))
  return (c_center,datac)

def shift_center(data,center):
  """
  Parameters
  ==========
  data is a list of frame with property "center"

  Return
  ======
  datas : shifted frames
  """
  cX,cY = center
  datas = []
  for f in data:
    f = f.copy()  # Important
    x,y = f["center"]
    dX = int(x - cX)
    dY = int(y - cY)
    print(dX,dY)
    # Shift every point with dX, dY
    top_lip = []
    for p in f["top_lip"]:
      x,y = p
      x += dX
      y += dY
      top_lip.append((x,y))  

    bottom_lip = []
    for p in f["bottom_lip"]:
      x,y = p
      x += dX
      y += dY
      bottom_lip.append((x,y))

    # print("Shifted----\n\t{}\n\t{}".format(f["top_lip"],top_lip))
    f["top_lip"] = top_lip
    f["bottom_lip"] = bottom_lip
    datas.append(f)

  return datas

def stabilize1(data):  
  """
  Stabilize method 1
  use four corners for ROI to find center point and transform all frames using difference.
  """
  c_center, datac = find_center(data)
  print("Clip center: {}".format(c_center))
  # print(datac[0]['center'])

  datas = shift_center(datac,c_center)

  print("Origin\n",data[0])
  print("Origin with center\n",datac[0])
  print("Shifted\n",datas[0])
  return datas

def stabilize2(data):
  """
  Use the left and right most edge of lips to find the center
  """
  datas = []
  return datas

def vec2viz(vector_file,shake_threshold=10):
  """
  shake_threshold is used to detemrine the distance to be ignored for detection.
  threshold in PERCENTAGE
  """
  _logger.info("Input file: {}".format(vector_file))
  import json
  with open(vector_file) as json_file:
    data = json.load(json_file)
    
    data.sort(key=lambda x : x["frame_id"])
    
    datas = stabilize1(data)


    # for f in datac:
    #   if f["frame_id"] == 0:
    #     continue
      # print("top_lip",len(f["top_lip"]))  #>>>> 12
      # print("bottom_lip",len(f["bottom_lip"])) #>>>> 12
      # TODO How to evaluate direction? angle of 45' ?
      
      # Every frame will shift to this center
      
      # for k in f.keys():
      #   print(k,f[k])

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
  vec2viz(args.vector_file)
  _logger.info("Script ends here")


def run():
  """Entry point for console_scripts
  """
  main(sys.argv[1:])


if __name__ == "__main__":
  run()
