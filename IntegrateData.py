import os
import sys

import pandas as pd
import numpy as np

import hdf5plugin
import fabio
import pyFAI
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve


def progress(count: int, total: int, status='') -> None:
    """
    Progress bar for sys

    :param count: Current count
    :type count: int
    :param total: Total counts
    :type total: int
    :param status: Optional status string to display
    :type status: str
    :return: None
    """
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %4.1f%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()


def baseline_als(y, lam, p, niter=10):
    """
    Baseline fit according to:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    :param y:
    :param lam:
    :param p:
    :param niter:
    :return:
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def gauss(x, *p):
    """
    Function for evaluating single gaussian at point(s) x

    :param x: Points to evaluate the gaussian at
    :type x: int or numpy.ndarray
    :param p: Gaussian description, p[0]: intensity, p[1]: center, p[2]: sigma
    :return: Value of the gaussian at the x points
    :rtype: int or numpy.ndarray
    """
    return p[0] * np.exp(-np.power(x - p[1], 2) / (2 * np.power(p[2], 2)))


def integrate_run(root, measurement_name, peak_fit=None, peak_guess=None):
    ai = pyFAI.load(os.path.join(root,
                                 measurement_name,
                                 measurement_name + '.poni'))
    try:
        mask_file = fabio.open(os.path.join(root,
                                            measurement_name,
                                            measurement_name + '_mask.edf'))
        mask = mask_file.data
    except FileNotFoundError:
        mask = None
    try:
        flatfield_file = fabio.open(os.path.join(root,
                                                 measurement_name,
                                                 measurement_name + '_flatfield.tiff'))
        flatfield = flatfield_file.data
        flatfield[flatfield > 1000] = 1
    except FileNotFoundError:
        flatfield = None

    with open(os.path.join(root, measurement_name, measurement_name + '.spec'), "r") as fi:
        _spec_data = []
        run = []
        names = None
        more_data = False
        for ln in fi:
            if ln.startswith('#L'):
                names = ln.split()[1:]
                run = []
                more_data = True
            elif ln.startswith('#S'):
                # save last run if exists
                if run and names:
                    _spec_data.append(pd.DataFrame(data=run[:-1], columns=names))
                    more_data = False
            elif ln.startswith('#C'):
                if run and names:
                    _spec_data.append(pd.DataFrame(data=run, columns=names))
                    more_data = False
                    run = []
            elif more_data:
                run.append(ln.split())
    if run:
        _spec_data.append(pd.DataFrame(data=run, columns=names))

    _all_patterns = []
    q = None
    for idx, run in enumerate(_spec_data):
        if 'eiger_data_filename' in run.columns:
            patterns = []
            for image in range(len(run)):
                progress(image, len(run), 'integrating run %d' % idx)
                path = os.path.join(root,
                                    measurement_name,
                                    'eiger',
                                    run['eiger_data_filename'][image] +
                                    '_data_%06d' % int(run['first_image_Nr'][image]))
                try:
                    file = fabio.open(path + '.h5')
                    data = file.data
                    data[data > 1e5] = 0
                    result = ai.integrate1d(data,
                                            npt=3000,
                                            unit="q_nm^-1",
                                            mask=mask,
                                            flat=flatfield)
                    bgr = baseline_als(result[1], 1e6, 0.01)
                    bgr = 0
                    res = result[1] - bgr
                    q = result[0]
                    patterns.append(res)
                except FileNotFoundError:
                    patterns.append(None)
            if q is None:
                df = None
            else:
                df = pd.DataFrame(patterns, columns=q)
                print(df)
            progress(len(run), len(run), 'run %d done' % idx)
            _all_patterns.append(df)
        else:
            _all_patterns.append(None)
    return _spec_data, _all_patterns


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('ERROR: Not enough input parameters.')
    elif len(sys.argv) > 3:
        sys.exit('ERROR: Too many input parameters.')
    else:
        spec_data, all_patterns = integrate_run(sys.argv[1], sys.argv[2])
        save_path = os.path.join(sys.argv[1], sys.argv[2], 'integrated_data')
        try:
            os.mkdir(save_path)
        except FileExistsError:
            sys.exit('ERROR: Output folder already exists.')
        for idx in range(len(spec_data)):
            if all_patterns[idx] is not None:
                all_patterns[idx].to_csv(os.path.join(save_path, sys.argv[2] + '_run%d_patterns.csv' % idx))
                spec_data[idx].to_csv(os.path.join(save_path, sys.argv[2] + '_run%d_metadata.csv' % idx))
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.pcolormesh(all_patterns[idx].values)
                fig.savefig(os.path.join(save_path, sys.argv[2] + '_run%d_heatmap.png' % idx),dpi=600)
