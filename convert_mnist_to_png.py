#!/usr/bin/env python

import os
import struct
import sys
import argparse

from array import array
from os import path

import png


# source: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
def read(dataset="training", path="."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir):
    # create output directories
    output_dirs = [path.join(output_dir, str(i)) for i in range(10)]
    for outdir in output_dirs:
        if not path.exists(outdir):
            os.makedirs(outdir)

    # write data
    print("\nWriting files to {}:".format(output_dir))
    no = len(labels)
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".png")
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [
                data[(i*rows*cols + j*cols): (i*rows*cols + (j+1)*cols)]
                for j in range(rows)]
            w.write(h, data_i)
        if i % 500 == 0:
            print("\rWriting files to {}.....{:.2f}%".format(output_dir, 100*i/no), end='')

    print("\rWriting files to {}.....Done               ".format(output_dir), end='')


if __name__ == "__main__":
    handler = argparse.ArgumentParser(description='Convert MNIST dataset to png format')
    handler.add_argument('input', help='MNIST directory')
    handler.add_argument('--output', metavar='', default='out', help='.png output directory')

    flags = handler.parse_args()

    input_path = flags.input
    output_path = flags.output

    for dataset in ["training", "testing"]:
        labels, data, size, rows, cols = read(dataset, input_path)
        write_dataset(labels, data, size, rows, cols,
                      path.join(output_path, dataset))