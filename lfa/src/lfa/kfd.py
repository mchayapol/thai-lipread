# -*- coding: utf-8 -*-


"""
Keyframe Detector
Input: LFA-CSV
Output: KFD-CSV


Original file is located at
    https://colab.research.google.com/drive/1K4faoIdDHiNNFTZhrdlGjdyT100H-az8

# Key Frame Detection
from CSV LFA 2021 to CSV Training Dataset  
LFA has min,max
"""


import argparse
import logging
import sys

import pandas as pd
import numpy as np
from mlr import util, profile_box

from lfa import __version__

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"

_logger = logging.getLogger(__name__)




def prepare_filenames(csvFilename):
    import os
    # csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.lfa.csv"
    filename, file_extension = os.path.splitext(csvFilename)
    # print(f"Name:\t{filename}\nExt:\t{file_extension}")

    csvFilename1kf = f"{filename}.1kf.csv"
    csvFilename3kf = f"{filename}.3kf.csv"
    return (csvFilename1kf, csvFilename3kf)


def appendColName(df, postfix):
    columns = {}
    # print(list(df.columns))
    for c in list(df.columns):
        columns[c] = f"{c}{postfix}"
    # print(columns)
    return df.rename(columns=columns)


def compute_keyframes(viseme, csvFilename):
    import pandas as pd
    import numpy as np

    """
  Compute keyframe 1 and 3
  and then export to 1kf.CSV and 3kf.CSV
  """

    csvFilename1kf, csvFilename3kf = prepare_filenames(csvFilename)

    df = pd.read_csv(csvFilename)
    df["viseme"] = viseme

    # df.columns
    # df[['frame#','sum','min','max','teeth_LAB','teeth_LUV']]
    # df[['frame#','sum','min','max','teeth_LAB','teeth_LUV']]

    df2 = df[['frame#', 'sum', 'min', 'max', 'teeth_LAB', 'teeth_LUV']]
    df2[['sum', 'min', 'max', 'teeth_LAB', 'teeth_LUV']].plot()

    df3 = df2.loc[df2['max'].notnull() | df2['min'].notnull()]

    """# Cleaning Up
  Assume that min and max values do not appear in the same row.
  """

    # Method 1: 3-Keyframe Data
    # df3 contains non-null min and max

    minFound = 0
    maxFound = 0
    maxR = minR = None
    L = []
    for index, row in df3.iterrows():
        # if row['max'] != float('nan'):
        if not np.isnan(row['max']):
            print('1---', row['max'], type(row['max']), row['max']
                  != float('nan'), np.isnan(row['max']))
            maxFound += 1
            minFound = 0
            maxR = row
            if maxFound == 1 and minFound == 0:
                if minR is not None:
                    L.append(minR)
                    print('minR', minR['min'], minR['max'])
            elif minFound == 0 and maxR['max'] < row['max']:
                maxR = row

        # if row['min'] != float('nan'):
        if not np.isnan(row['min']):
            print('2---', row['min'], type(row['min']), row['min']
                  != float('nan'), np.isnan(row['min']))
            minFound += 1
            maxFound = 0
            minR = row
            if minFound == 1 and maxFound == 0:
                if maxR is not None:
                    L.append(maxR)
                    print('maxR', maxR['min'], maxR['max'])
            elif maxFound == 0 and minR['min'] > row['min']:
                minR = row

    df5 = pd.DataFrame(L)
    # df5[['min','max']]
    # df5.columns
    # df5

    """## Use frame# to select midpoint between min-max and max-min."""

    L = []
    ticker = 0
    firstRow = True

    for index, row in df5.iterrows():
        fno = int(row['frame#'])
        if firstRow:
            firstRow = False
            # print(row)
            if np.isnan(row['min']):  # there could be chance that the first row is not min, skip it
                print(f"Skip first row {fno}")
                continue

        # print(f"{fno} ticker={ticker}")
        if ticker == 0:
            # print(row)
            if np.isnan(row['min']):
                raise Exception("Assertion error: expect min")
            minfno1 = fno
        if ticker == 1:
            if np.isnan(row['max']):
                raise Exception("Assertion error: expect max")
            maxfno = fno
            midfno1 = int((minfno1 + maxfno) / 2)
            L.append(midfno1)
            L.append(maxfno)
            # print(midfno1,maxfno)
        if ticker == 2:
            if np.isnan(row['min']):
                raise Exception("Assertion error: expect min")
            minfno1 = fno
            minfno2 = fno
            midfno2 = int((minfno2 + maxfno) / 2)
            L.append(midfno2)
            # print(midfno2)
            ticker = 0
        ticker += 1

    # L

    # print(L[0:3])
    # print(L[3:6])
    # print(L[6:9])

    f1 = df[df['frame#'] == 30].drop(['Unnamed: 0'], axis=1).reset_index()
    f2 = df[df['frame#'] == 38].drop(['Unnamed: 0'], axis=1).reset_index()
    f3 = df[df['frame#'] == 44].drop(['Unnamed: 0'], axis=1).reset_index()

    f1 = appendColName(f1, 'a')
    f2 = appendColName(f2, 'b')
    f3 = appendColName(f3, 'c')
    # print(f1)

    f = pd.concat([f1, f2, f3], axis=1)

    print(len(L))
    samples = int(len(L)/3)
    L3 = []
    for i in range(samples):
        fnos = L[i*3:i*3+3]
        # print(fnos)
        f1 = df[df['frame#'] == fnos[0]].drop(['Unnamed: 0'], axis=1).reset_index()
        f2 = df[df['frame#'] == fnos[1]].drop(['Unnamed: 0'], axis=1).reset_index()
        f3 = df[df['frame#'] == fnos[2]].drop(['Unnamed: 0'], axis=1).reset_index()

        f1 = appendColName(f1, 'a')
        f2 = appendColName(f2, 'b')
        f3 = appendColName(f3, 'c')

        f = pd.concat([f1, f2, f3], axis=1)
        # print(f)
        ser = f.iloc[0]
        L3.append(ser)
    # L3

    print(len(L3))
    df3kf = pd.DataFrame(L3).reset_index().drop(['index'], axis=1)
    # df3kf

    """## Export to CSV 3kf"""

    df3kf.to_csv(csvFilename3kf, index=False)
    print(f"Export 3KF to {csvFilename3kf}")

    """# Method 2: 1-Keyframe Data (MAX only)"""

    df6 = df5[~np.isnan(df5['max'])]
    df1kf = df[df['frame#'].isin(df6['frame#'])].drop(['Unnamed: 0'], axis=1)
    # df1kf

    """## Export to CSV 1kf"""

    df1kf.to_csv(csvFilename1kf, index=False)
    print(f"Export 1KF to {csvFilename1kf}")

    return (df3kf, df1kf)


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
        version="pb {ver}".format(ver=__version__),
    )

    parser.add_argument(dest="viseme",
                        help="Viseme", type=str, metavar="VISEME")

    parser.add_argument(dest="csv_filename",
                        help="Lip-Geometry CSV filename", type=str, metavar="INPUT_CSV")

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
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    # global csv_filename
    # global viseme

    args = parse_args(args)
    setup_logging(args.loglevel)
    csv_filename = args.csv_filename
    viseme = args.viseme
    df3kf, df1kf = compute_keyframes(viseme,csv_filename)


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
    # viseme = "v21"
    # csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.lfa.csv"
    # df3kf, df1kf = compute_keyframes(viseme,csvFilename)
