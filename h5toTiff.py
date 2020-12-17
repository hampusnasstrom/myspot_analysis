from typing import List

import hdf5plugin  # Needed to decompress the LZ4 compression
import fabio
import os
import tifffile
import numpy as np


def h5s_to_average_tiff(paths: List[str], threshold=None):
    data = 0
    for path in paths:
        file = fabio.open(path)
        single_data = file.data
        if threshold:
            single_data[single_data > threshold] = -2
        data += single_data
    data = data / len(paths)
    tifffile.imwrite(paths[0][:-22] + '_averaged.tiff', data)


def h5_to_tiff(path: str, threshold=None):
    file = fabio.open(path)
    data = file.data
    if threshold:
        data[data > threshold] = -2
    tifffile.imwrite(path[:-3] + '_masked.tiff', data)


if __name__ == "__main__":
    root = r'\\ul-nas\myspot_data\2020-10-13-Naessstroem'
    measurement_name = '2020-10-18_flatfield-InkCube'
    run = 1
    images = np.arange(1, 61)
    run_paths = []
    for image in images:
        run_paths.append(os.path.join(root,
                                      measurement_name,
                                      'eiger',
                                      measurement_name + '_%06d_data_%06d.h5' % (run, image)))
    h5s_to_average_tiff(paths=run_paths, threshold=1e9)
