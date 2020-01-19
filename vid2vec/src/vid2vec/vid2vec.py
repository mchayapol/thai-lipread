# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

  console_scripts =
        fibonacci = vid2vec.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys
import logging

from vid2vec import __version__

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

def vid2vec(v):
  """Main entry for vid2vec function. will generate a JSON file of property vector of the given video

  Args:
    v (str): string

  Returns:
    int: -1 video does not exist
  """  
  try:
    with open(v) as f:
      # Do something with the file
      pass
  except IOError:
    _logger.warn('File "{}" not accessible'.format(v))
    return -1

  return v

def parse_args(args):
  """Parse command line parameters

  Args:
    args ([str]): command line parameters as list of strings

  Returns:
    :obj:`argparse.Namespace`: command line parameters namespace
  """
  parser = argparse.ArgumentParser(
      description="Generate a vector file of a video clip.")
  parser.add_argument(
      "--version",
      action="version",
      version="vid2vec {ver}".format(ver=__version__))
  parser.add_argument(
      dest="v",
      help="video filename",
      type=str,
      metavar="STRING")        
  # parser.add_argument(
  #     dest="n",
  #     help="n-th Fibonacci number",
  #     type=int,
  #     metavar="INT")
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
  ret = vid2vec(args.v)
  print("RET {}".format(ret))
  _logger.info("Script ends here")


def run():
  """Entry point for console_scripts
  """
  main(sys.argv[1:])


if __name__ == "__main__":
  run()
