"""
Convert processed CSV (.lfa.csv) to ARFF
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


def format_row(df, index, step):
    columns = ['teeth_LAB', 'teeth_LUV', '49_angle',
               '50_angle', '51_angle', '52_angle', '53_angle', '54_angle', '55_angle',
               '56_angle', '57_angle', '58_angle', '59_angle', '61_angle', '62_angle',
               '63_angle', '64_angle', '65_angle', '66_angle', '67_angle', '68_angle',
               '69_angle', '70_angle', '71_angle']
    S = df.loc[index, columns]
    S = pd.DataFrame(S)

    # print(S)
    formatted_columns = []
    for c in S.index:
        formatted_columns.append("{}_{}".format(step, c))
    S.index = formatted_columns
    S = S.transpose().reset_index().drop(columns=['index'])
    # print(S)
    return S


def prepare_frames(df, viseme_class):
    gap = 5

    # MAX List of frame index
    max_index = df['max'].loc[df['max'].notnull()].index
    sample_count = 0
    data = None
    for B in max_index:
        A = B - gap
        C = B + gap
        if A < 0 or C > df.shape[0]:
            print("- Out of bound {},{},{}".format(A, B, C))
            continue

        sA = format_row(df, A, 'A')
        sB = format_row(df, B, 'B')
        sC = format_row(df, C, 'C')
        # row = sA.append(sB).append(sC)

        row = pd.concat([sA, sB, sC], axis=1, sort=False)
        row['class'] = viseme_class
        # print(row)
        if data is None:
            data = pd.DataFrame(row)
        else:
            data = data.append(row)
        sample_count += 1
        # print(data)

    # data
    print('= {} samples'.format(sample_count))
    print(data.shape)
    return data
# arffFile = vfile+'.arff'
# print("Output ARFF "+arffFile)
# arff.dump(vfile+'.arff', data.values, relation='visemes', names=data.columns)


def process_maybe():
    # vfile = 'v03'
    # viseme_class = '3'
    # df = pd.read_csv(vfile+".processed.csv")
    data = None
    for i in range(len(files)):
        vfile = files[i]
        viseme_class = classes[i]
        print("Processing {}".format(vfile))
        df = pd.read_csv(vfile+".processed.csv")
        d = prepare_frames(df, viseme_class)
        if data is None:
            data = pd.DataFrame(d)
            print("Create new DataFrame")
        else:
            data = data.append(d)
            print("Append data", data.shape, d.shape)

    print(data.shape)
    arffFile = 'visemes.arff'
    print("Output ARFF "+arffFile)
    arff.dump(arffFile, data.values, relation='visemes', names=data.columns)


def filter_columns(df0):
    # files = ['v01','v02','v03','v04','v05','v06','v07',
    #          'v08','v09','v10','v11','v12','v13','v14',
    #          'v15','v16','v17','v18','v19','v20','v21']
    # classes = ['1','2','3','4','5','6','7',
    #            '8','9','10','11','12','13','14',
    #            '15','16','17','18','19','20','21']
    """
    Data Selection
    - Only point 49 - 72
    - Use 49 as corner to [50,51,52,58,59,60,61,62,63,67,68]
    - Use 55 as corner to [52,53,54,56,57,58,63,64,65,66,67]
    - and finall 49 and 45 for ROI tilt
    """
    df = df0[["frame#",
              "50_angle",
              "52_angle",
              "54_angle",
              "65_angle",
              "68_angle",
              "teeth_LAB",
              "teeth_LUV"
              ]]
    return df

def process_rows(df0):
    df0['teeth'] = df0.teeth_LAB + df0.teeth_LUV
    return df0

def filter_rows(df0):
    df = df0.loc[(df0['teeth_LAB'] > 0) & (df0['teeth_LUV'] > 0)]
    return df

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
    parser.add_argument("-ff", dest="frame_first",help="First Frame", type=int, default=1)
    parser.add_argument("-fl", dest="frame_last",help="Last Frame (default: last frame of the input clip)", type=int, default=-1)

    parser.add_argument(dest="csv_filename",
                        help="Lip-Geometry CSV filename (.lfa.csv)", type=str, metavar="CSV")

    parser.add_argument(
        "-q",
        dest="disable_viz",
        help="Disable visualisation",
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
    global frame_scope
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    csv_filename = args.csv_filename
    frame_scope = (args.frame_first,args.frame_last)
    if ".lfa." not in csv_filename:
        print("Not LFA file? Consider adding .lfa.csv extension")

    print("Preparing ARFF from {}".format(csv_filename))
    print("Frame Scope: ",frame_scope)
    df = pd.read_csv(csv_filename)
    df1 = filter_columns(df)
    df2 = process_rows(df1)
    
    df3 = filter_rows(df2)
    print(df3.head(40))
    # df2 = select_uttering(df1)

    # dfa.to_csv(csv_filename.replace(".csv", "")+f'-m{method}.lfa.csv')

    # arff.dump(csv_filename.replace(".csv", "")+f'-m{method}.arff',
    #           dfa.values,
    #           relation='relation name',
    #           names=dfa.columns)

    # _logger.info("Script ends here")


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
