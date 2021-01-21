# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = lfa.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arff

from lfa import __version__

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
    for i in range(n - 1):
        a, b = b, a + b
    return a


def map_quadrant(d):
    if 90 < d <= 180:
        d2 = d - 180
    elif d < -90:
        d2 = d + 180
    else:
        d2 = d
    return d2


def filter_columns(raw):
    """
    Data Selection
    - Only point 49 - 72
    - Use 49 as corner to [50,51,52,58,59,60,61,62,63,67,68]
    - Use 55 as corner to [52,53,54,56,57,58,63,64,65,66,67]
    - and finall 49 and 45 for ROI tilt
    """
    df = raw[["frame#",
              "49_top_lip_x",
              "49_top_lip_y",
              "50_top_lip_x",
              "50_top_lip_y",
              "51_top_lip_x",
              "51_top_lip_y",
              "52_top_lip_x",
              "52_top_lip_y",
              "53_top_lip_x",
              "53_top_lip_y",
              "54_top_lip_x",
              "54_top_lip_y",
              "55_top_lip_x",
              "55_top_lip_y",
              "56_top_lip_x",
              "56_top_lip_y",
              "57_top_lip_x",
              "57_top_lip_y",
              "58_top_lip_x",
              "58_top_lip_y",
              "59_top_lip_x",
              "59_top_lip_y",
              "60_top_lip_x",
              "60_top_lip_y",
              "61_bottom_lip_x",
              "61_bottom_lip_y",
              "62_bottom_lip_x",
              "62_bottom_lip_y",
              "63_bottom_lip_x",
              "63_bottom_lip_y",
              "64_bottom_lip_x",
              "64_bottom_lip_y",
              "65_bottom_lip_x",
              "65_bottom_lip_y",
              "66_bottom_lip_x",
              "66_bottom_lip_y",
              "67_bottom_lip_x",
              "67_bottom_lip_y",
              "68_bottom_lip_x",
              "68_bottom_lip_y",
              "69_bottom_lip_x",
              "69_bottom_lip_y",
              "70_bottom_lip_x",
              "70_bottom_lip_y",
              "71_bottom_lip_x",
              "71_bottom_lip_y",
              "72_bottom_lip_x",
              "72_bottom_lip_y"
              ]]
    return df


def method0(raw):
    df = filter_columns(raw)
    corner_y = '49_top_lip_y'
    corner_x = '49_top_lip_x'

    dfa = pd.DataFrame()
    # dfa[['frame#', 'teeth_LAB', 'teeth_LUV']] = raw[['frame#', 'teeth_LAB', 'teeth_LUV']]
    for pno in range(49, 60, 1):
        y1 = '{}_top_lip_y'.format(pno+1)
        x1 = '{}_top_lip_x'.format(pno+1)
        y0 = corner_y
        x0 = corner_x
        a_label = '{}_angle'.format(pno)
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series

    for pno in range(61, 72, 1):
        y1 = '{}_bottom_lip_y'.format(pno+1)
        x1 = '{}_bottom_lip_x'.format(pno+1)
        y0 = corner_y
        x0 = corner_x
        a_label = '{}_angle'.format(pno)
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series

    return dfa


"""
Method 1: angles between adjacent points.
"""


def method1(raw):
    df = filter_columns(raw)
    dfa = pd.DataFrame()
    # dfa['frame#'] = raw['frame#']
    dfa[['frame#', 'teeth_LAB', 'teeth_LUV']] = raw[['frame#', 'teeth_LAB', 'teeth_LUV']]
    for pno in range(49, 60, 1):
        y1 = '{}_top_lip_y'.format(pno+1)
        x1 = '{}_top_lip_x'.format(pno+1)
        x0 = '{}_top_lip_x'.format(pno)
        y0 = '{}_top_lip_y'.format(pno)
        a_label = '{}_angle'.format(pno)
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series

    for pno in range(61, 72, 1):
        y1 = '{}_bottom_lip_y'.format(pno+1)
        x1 = '{}_bottom_lip_x'.format(pno+1)
        y0 = '{}_bottom_lip_y'.format(pno)
        x0 = '{}_bottom_lip_x'.format(pno)
        a_label = '{}_angle'.format(pno)
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series
    return dfa


def method2(raw):
    df = filter_columns(raw)
    top_pairs = [(49, 55),
                 (49, 50), (49, 51), (49, 52), (55, 54), (55, 53), (55, 52),
                 (49, 59), (49, 58), (55, 57), (55, 58)]
    bottom_pairs = [(67, 66), (67, 65), (67, 64), (61, 62), (61, 63), (61, 64),
                    (67, 69), (67, 70), (61, 71), (61, 70)
                    ]

    dfa = pd.DataFrame()
    # dfa['frame#'] = raw['frame#']
    dfa[['frame#', 'teeth_LAB', 'teeth_LUV']] = raw[['frame#', 'teeth_LAB', 'teeth_LUV']]

    for p in top_pairs:
        y1 = '{}_top_lip_y'.format(p[1])
        x1 = '{}_top_lip_x'.format(p[1])
        x0 = '{}_top_lip_x'.format(p[0])
        y0 = '{}_top_lip_y'.format(p[0])
        a_label = '{}_{}_angle'.format(p[0], p[1])
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series

    for p in bottom_pairs:
        y1 = '{}_bottom_lip_y'.format(p[1])
        x1 = '{}_bottom_lip_x'.format(p[1])
        y0 = '{}_bottom_lip_y'.format(p[0])
        x0 = '{}_bottom_lip_x'.format(p[0])
        a_label = '{}_{}_angle'.format(p[0], p[1])
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series
    return dfa


def method3(raw):
    df = filter_columns(raw)
    pairs = [
        ('49_top_lip', '50_top_lip'),
        ('49_top_lip', '51_top_lip'),
        ('49_top_lip', '52_top_lip'),
        ('49_top_lip', '61_bottom_lip'),
        ('49_top_lip', '62_bottom_lip'),
        ('49_top_lip', '63_bottom_lip'),
        ('49_top_lip', '60_top_lip'),
        ('49_top_lip', '67_bottom_lip'),
        ('49_top_lip', '68_bottom_lip'),
        ('49_top_lip', '58_top_lip'),
        ('49_top_lip', '59_top_lip'),
        ('49_top_lip', '60_top_lip')
    ]

    dfa = pd.DataFrame()
    # dfa['frame#'] = raw['frame#']
    dfa[['frame#', 'teeth_LAB', 'teeth_LUV']] = raw[['frame#', 'teeth_LAB', 'teeth_LUV']]

    for p in pairs:
        y1 = '{}_y'.format(p[1])
        x1 = '{}_x'.format(p[1])
        x0 = '{}_x'.format(p[0])
        y0 = '{}_y'.format(p[0])
        a_label = '{}_{}_angle'.format(p[0], p[1])
        # print('{} = {}-{} / {}-{}'.format(a_label,y1,y0,x1,x0))
        series = np.degrees(np.arctan2(
            df[y1] - df[y0], df[x1] - df[x0])).apply(map_quadrant)
        dfa[a_label] = series
    return dfa


def viz(dfa, raw,to_file_only=False):
    text_kwargs = dict(ha='left', va='top', fontsize=10, color='tab:gray')
    fig, axs = plt.subplots(5, sharex=True, sharey=True, gridspec_kw={'hspace': 0},figsize=(20,10))
    # fig = plt.figure()
    fig.suptitle("Viseme {}".format(csv_filename), fontsize=16)
    fig.text(0.5, 0.04, 'Frame#', ha='center')
    fig.text(0.04, 0.5, 'Angles (degrees)', va='center', rotation='vertical')

    # 1. SUM
    a_cols = [col for col in dfa.columns if 'angle' in col]
    dfa["sum"] = dfa[a_cols].sum(axis=1)

    ax = axs[0]
    # ax = plt.subplot(511)
    # ax = plt.subplot(511,sharex=True)
    # ax.set_title('SUM')
    # ax.text(0, 0, 'SUM', **text_kwargs)
    fig.text(0.13, 0.85, 'SUM',fontsize=14,color='gray')
    # ax.text(0, 2, 'SUM',verticalalignment='top')
    # ax.set_ylabel('Degrees')
    # ax.set_xlabel('Frame#')
    ax.plot(dfa.index, dfa['sum'])
    # fig.show()

    # 2. ABS SUM
    a_cols = [col for col in dfa.columns if 'angle' in col]
    dfa["sum"] = dfa[a_cols].abs().sum(axis=1)
    # ax = plt.subplot(512,sharex=True)
    ax = axs[1]
    # ax.set_title('ABS SUM')
    fig.text(0.13, 0.7, 'ABS SUM',fontsize=14,color='gray')
    # ax.set_ylabel('Degrees')
    # ax.set_xlabel('Frame#')

    # ax.figure(figsize=(20, 10))
    ax.plot(dfa.index, dfa['sum'])
    # fig.show()

    # 3. MEAN
    a_cols = [col for col in dfa.columns if 'angle' in col]
    dfa["sum"] = dfa[a_cols].mean(axis=1)
    ax = axs[2]
    # ax = plt.subplot(513)
    # ax.set_title('MEAN')
    fig.text(0.13, 0.55, 'MEAN',fontsize=14,color='gray')
    # ax.set_ylabel('Degrees')
    # ax.set_xlabel('Frame#')
    ax.plot(dfa.index, dfa['sum'])
    # fig.show()

    # 3-- MEAN MIN/MAX SIGNAL
    from scipy.signal import argrelextrema
    # Generate a noisy AR(1) sample
    n = 10  # number of points to be checked before and after
    # Find local peaks
    dfa['min'] = dfa["sum"].iloc[argrelextrema(
        dfa["sum"].values, np.less_equal, order=n)[0]]
    dfa['max'] = dfa["sum"].iloc[argrelextrema(
        dfa["sum"].values, np.greater_equal, order=n)[0]]

    # Plot results
    # ax = plt.subplot(514)
    ax = axs[3]
    # ax.set_title('SIGNAL teeth LAB')
    fig.text(0.13, 0.39, 'SIGNAL teeth LAB',fontsize=14,color='gray')
    # ax.set_ylabel('Degrees')
    # ax.set_xlabel('Frame#')
    ax.scatter(dfa.index, dfa['min'], c='g')
    ax.scatter(dfa.index, dfa['max'], c='r')
    ax.plot(raw.index, raw['teeth_LAB'], c='y')
    # plt.plot(raw.index, raw['teeth_LUV'], c='g')
    ax.plot(dfa.index, dfa['sum'])

    # fig.show()

    # 3-- MEAN MIN/MAX
    # ax = plt.subplot(515)
    ax = axs[4]
    # ax.set_title('SIGNAL teeth LUV')
    fig.text(0.13, 0.23, 'SIGNAL teeth LUV',fontsize=14,color='gray')
    # ax.set_ylabel('Degrees')
    # ax.set_xlabel('Frame#')
    ax.scatter(dfa.index, dfa['min'], c='r')
    ax.scatter(dfa.index, dfa['max'], c='g')
    ax.plot(raw.index, raw['teeth_LUV'], c='g')
    ax.plot(dfa.index, dfa['sum'])


    if to_file_only:
        fig_filename = csv_filename.replace('.csv','')+'.png'
        # fig_filename = csv_filename.replace('.csv','')+'.pdf'   # vectorised figure
        print(f"Save figure to {fig_filename}")
        # plt.figure(figsize=(40, 20))        
        plt.savefig(fig_filename, bbox_inches='tight')
    else:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()



def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Lip Features Analysis")
    parser.add_argument(
        "--version",
        action="version",
        version="lfa {ver}".format(ver=__version__),
    )
    parser.add_argument("-m", dest="method",
                        help="Method (default 0)", type=int, default=0)
    parser.add_argument(dest="csv_filename",
                        help="Lip-Geometry CSV filename", type=str, metavar="CSV")

    parser.add_argument(
        "-q",
        dest="disable_viz",
        help="Disable visualisation",
        action="store_const",
        const=True,
    )

    parser.add_argument(
        "-s",
        dest="save_fig",
        help="Save visualisation figure",
        action="store_const",
        const=True,
    )


    # Log related arguments
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
        filename='lfa.log',
        filemode='a',
        level=loglevel, 
        # stream=sys.stdout, 
        format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    global csv_filename
    global method
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    csv_filename = args.csv_filename
    method = args.method
    _logger.info("Analysis Method {}".format(method))

    raw_df = pd.read_csv(csv_filename)
    dfa = {
        0: method0,
        1: method1,
        2: method2,
        3: method3
    }[args.method](raw_df)

    # viz_mean_min_max(dfa,raw_df)
    if not args.disable_viz:
        viz(dfa, raw_df)
    elif args.save_fig:
        viz(dfa, raw_df, True)

    dfa.to_csv(csv_filename.replace(".csv", "")+f'-m{method}.lfa.csv')

    arff.dump(csv_filename.replace(".csv", "")+f'-m{method}.arff',
              dfa.values,
              relation='relation name',
              names=dfa.columns)

    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
