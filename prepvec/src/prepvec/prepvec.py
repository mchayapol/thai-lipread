# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

  console_scripts =
     fibonacci = prepvec.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import logging
import sys

from prepvec import __version__

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"

_logger = logging.getLogger(__name__)

def flatten_json(jsondata):
  out = {}
  jsondata_flat = []

  def flatten(x, name=''):
    if type(x) is dict:
      for a in x:
        flatten(x[a], name + a + '_')
    elif type(x) is list:
      i = 0
      for a in x:
        flatten(a, name + str(i) + '_')
        i += 1
    else:
      out[name[:-1]] = x

  for r in jsondata:
    flatten(r)
    jsondata_flat.append(out)
    out = {}
  # flatten(jsondata)
  # return out
  # print(jsondata_flat)
  return jsondata_flat

def prepvec(jsonfile):
  import json
  import csv
  import os

  csvfile = jsonfile.replace('.json','.csv')
  # print("Hello "+jsonfile)

  # import pandas as pd
  # df = pd.read_json (jsonfile)
  # df.to_csv (jsonfile+".csv", index = None)
  # return True
  with open(jsonfile) as json_file:
      data = json.load(json_file)
      jsondata_flat = flatten_json(data)
      # print(jsondata_flat)
      f = open(csvfile, "w", newline='')
      csv_writer = csv.writer(f) 
      
      # print(len(jsondata_flat.keys()))
      count = 0  
      for r in jsondata_flat: 
        # print(r)
        if count == 0:       
          # Writing headers of CSV file 
          header = r.keys() 
          csv_writer.writerow(header) 
          count += 1
      
        # Writing data of CSV file 
        csv_writer.writerow(r.values()) 
        _logger.debug(r.values())
       
      
      f.close()
      print("Saved to "+csvfile)
      # print(data[0].keys())
      # ['frame_id', 'top_lip', 'bottom_lip', 'teeth_appearance']

      # current_dir = os.getcwd()
      # basename = os.path.basename(v)
      # sep = os.path.sep
      # ofn = "{}{}{}.json".format(current_dir,sep,basename)
      # _logger.debug("\tOutput to "+ofn)
      # extract_features(v,ofn)




      # frame#,1_chin_x,1_chin_y,2_chin_x,2_chin_y,3_chin_x,3_chin_y,4_chin_x,4_chin_y,5_chin_x,5_chin_y,6_chin_x,6_chin_y,7_chin_x,7_chin_y,8_chin_x,8_chin_y,9_chin_x,9_chin_y,10_chin_x,10_chin_y,11_chin_x,11_chin_y,12_chin_x,12_chin_y,13_chin_x,13_chin_y,14_chin_x,14_chin_y,15_chin_x,15_chin_y,16_chin_x,16_chin_y,17_chin_x,17_chin_y,18_left_eyebrow_x,18_left_eyebrow_y,19_left_eyebrow_x,19_left_eyebrow_y,20_left_eyebrow_x,20_left_eyebrow_y,21_left_eyebrow_x,21_left_eyebrow_y,22_left_eyebrow_x,22_left_eyebrow_y,23_right_eyebrow_x,23_right_eyebrow_y,24_right_eyebrow_x,24_right_eyebrow_y,25_right_eyebrow_x,25_right_eyebrow_y,26_right_eyebrow_x,26_right_eyebrow_y,27_right_eyebrow_x,27_right_eyebrow_y,28_nose_bridge_x,28_nose_bridge_y,29_nose_bridge_x,29_nose_bridge_y,30_nose_bridge_x,30_nose_bridge_y,31_nose_bridge_x,31_nose_bridge_y,32_nose_tip_x,32_nose_tip_y,33_nose_tip_x,33_nose_tip_y,34_nose_tip_x,34_nose_tip_y,35_nose_tip_x,35_nose_tip_y,36_nose_tip_x,36_nose_tip_y,37_left_eye_x,37_left_eye_y,38_left_eye_x,38_left_eye_y,39_left_eye_x,39_left_eye_y,40_left_eye_x,40_left_eye_y,41_left_eye_x,41_left_eye_y,42_left_eye_x,42_left_eye_y,43_right_eye_x,43_right_eye_y,44_right_eye_x,44_right_eye_y,45_right_eye_x,45_right_eye_y,46_right_eye_x,46_right_eye_y,47_right_eye_x,47_right_eye_y,48_right_eye_x,48_right_eye_y,49_top_lip_x,49_top_lip_y,50_top_lip_x,50_top_lip_y,51_top_lip_x,51_top_lip_y,52_top_lip_x,52_top_lip_y,53_top_lip_x,53_top_lip_y,54_top_lip_x,54_top_lip_y,55_top_lip_x,55_top_lip_y,56_top_lip_x,56_top_lip_y,57_top_lip_x,57_top_lip_y,58_top_lip_x,58_top_lip_y,59_top_lip_x,59_top_lip_y,60_top_lip_x,60_top_lip_y,61_bottom_lip_x,61_bottom_lip_y,62_bottom_lip_x,62_bottom_lip_y,63_bottom_lip_x,63_bottom_lip_y,64_bottom_lip_x,64_bottom_lip_y,65_bottom_lip_x,65_bottom_lip_y,66_bottom_lip_x,66_bottom_lip_y,67_bottom_lip_x,67_bottom_lip_y,68_bottom_lip_x,68_bottom_lip_y,69_bottom_lip_x,69_bottom_lip_y,70_bottom_lip_x,70_bottom_lip_y,71_bottom_lip_x,71_bottom_lip_y,72_bottom_lip_x,72_bottom_lip_y,teeth_LAB,teeth_LUV

      # for p in data['people']:
      #     print('Name: ' + p['name'])
      #     print('Website: ' + p['website'])
      #     print('From: ' + p['from'])
      #     print('')
      
  
def fib(n):
  """Fibonacci example function

  Args:
    n (int): integer

  Returns:
    int: n-th Fibonacci number
  """
  assert n > 0
  a, b = 1, 1
  for i in range(n - 1):
    a, b = b, a + b
  return a


def parse_args(args):
  """Parse command line parameters

  Args:
    args ([str]): command line parameters as list of strings

  Returns:
    :obj:`argparse.Namespace`: command line parameters namespace
  """
  parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
  parser.add_argument(
    "--version",
    action="version",
    version="prepvec {ver}".format(ver=__version__),
  )
  # parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
  parser.add_argument(
    dest="jsonfile", help="JSON filename", type=str, metavar="JSONFILE")
  parser.add_argument(
    "-v",
    "--verbose",
    dest="loglevel",
    help="set loglevel to INFO",
    action="store_const",
    const=logging.INFO,
  )
  parser.add_argument(
    "-vv",
    "--very-verbose",
    dest="loglevel",
    help="set loglevel to DEBUG",
    action="store_const",
    const=logging.DEBUG,
  )
  return parser.parse_args(args)


def setup_logging(loglevel):
  """Setup basic logging

  Args:
    loglevel (int): minimum loglevel for emitting messages
  """
  logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
  logging.basicConfig(
    level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
  )


def main(args):
  """Main entry point allowing external calls

  Args:
    args ([str]): command line parameter list
  """
  args = parse_args(args)
  setup_logging(args.loglevel)
  # _logger.debug("Starting crazy calculations...")
  # print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
  prepvec(args.jsonfile)
  # _logger.info("Script ends here")


def run():
  """Entry point for console_scripts"""
  main(sys.argv[1:])


if __name__ == "__main__":
  run()
