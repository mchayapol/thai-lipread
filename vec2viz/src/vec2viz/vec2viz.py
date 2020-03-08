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

def vec2viz(vector_file):
  _logger.info("Input file: {}".format(vector_file))
  import json
  with open(vector_file) as json_file:
    data = json.load(json_file)
    data.sort(key=lambda x : x["frame_id"])
    for f in data:
      if f["frame_id"] == 0:
        continue
      # print("top_lip",len(f["top_lip"]))  >>>> 12
      # print("bottom_lip",len(f["bottom_lip"])) >>>> 12
      # TODO How to evaluate direction? angle of 45' ?
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
