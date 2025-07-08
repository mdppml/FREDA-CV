"""
This file is adapted from [Wenda: Weighted Elastic Net for Unsupervised Domain Adaptation].
Original Repository: https://github.com/PfeiferLabTue/wenda
License: GNU General Public License v3.0 (GPL-3.0)
"""

import numpy as np


def printTestErrors(pred_raw, test_y_raw, heading=None, indent=0):
    prefix = " " * indent
    errors = np.abs(test_y_raw - pred_raw)
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    corr = np.corrcoef(pred_raw, test_y_raw)[0, 1]
    std = np.std(errors)
    q75, q25 = np.percentile(errors, [75, 25])
    iqr = q75 - q25
    if heading is not None:
        print(prefix + heading)
        print(prefix + len(heading) * '-')
    print(prefix + "Mean abs. error:", mean_err)
    print(prefix + "Median abs. error:", median_err)
    print(prefix + "Correlation:", corr)
    print()
    return [mean_err, median_err, corr, std, iqr]
