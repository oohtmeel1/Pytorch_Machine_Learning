#!/usr/bin/env python3
""" LOADING FILES AND GETTING TRAIN TEST VAL
this is super specific. the items must be in train test val
and the files can get overwritten"""
import argparse
from typing import NamedTuple
import os
import pandas as pd


class Args(NamedTuple):
    """ Command-line arguments """
    file: str
    dir_: str
    overwrite: str
# --------------------------------------------------


def get_args():
    """ Get command-line arguments """

    parser = argparse.ArgumentParser(
        description='Make lists of labels for train, test, val',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', metavar='str',
                        help='''enter folder names,
                        for example train, test, val''', nargs='?')
    parser.add_argument('dir_', metavar='str',
                        help='specify directory',
                        nargs='*', default=os.getcwd())
    parser.add_argument('overwrite', metavar='str',
                        help='''specify whether to
                        overwrite current files''', nargs='*', default=False)
    args = parser.parse_args()

    return Args(args.file, args.dir_, args.overwrite)
# --------------------------------------------------
# Go to train test split folders


def main():
    """ Trying to load some files """
    args = get_args()
    file = args.file
    dir_ = args.dir_
    files = ['train', 'test', 'val']
    if file in files:
        print('folder selected : ', file)
    if file not in files:
        print('please specify a folder from one of train, test, val')
    if dir_ == os.getcwd():
        print('''Directory not specified,
            resulting file will be in:''', os.getcwd())
    else:
        dir_ = args.dir_

    def making_labels_now(file, dir_):
        starter_path = os.path.join(dir_, file)
        next_path = os.listdir(starter_path)
        return next_path
    j = making_labels_now(file, dir_)
    print("There are :", len(j), "files in this folder")

    def labels_made(j):
        data_labels_tr = []
        for i in j:
            data_labels_tr.append(i)
        partial_string = 'og'
        xy1 = pd.DataFrame(data_labels_tr)
        xy1['status'] = xy1[0].str.contains(partial_string).astype(int)
        xy1.columns = ['id', 'status']
        my_file = os.path.join(args.dir_, f"{file}_labels_final.csv")
        if os.path.isfile(my_file):
            print('file exists')
            l_ = input('Do you want to overwrite the labels? (yes or no),')
            if l_ == 'yes':
                xy1.to_csv(f"{file}_labels_final.csv", index=False)
                print('labels overwritten')
            if l_ == 'no':
                xy1.to_csv(f"{file}_labels_final.csv", index=False, mode='x')
                print('please move or rename file')
        else:
            print('new_file_made')
            xy1.to_csv(f"{file}_labels_final.csv", index=False)

    labels_made(j)


# --------------------------------------------------

if __name__ == '__main__':
    main()
