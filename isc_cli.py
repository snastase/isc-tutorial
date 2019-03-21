#!/usr/bin/env python3

import sys
import argparse
import logging
from glob import glob
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, zscore


# Set up logger first
logger = logging.getLogger(__name__)


# Set up argument parser
def parse_arguments(args):
    
    parser = argparse.ArgumentParser(
        description=("Python-based command-line program for computing "
                     "leave-one-out intersubject correlations (ISCs)"),
        epilog=("""
    This program provides a simple Python-based command-line interface (CLI) for
    running intersubject correlation (ISC) analysis. ISCs are computed using the
    leave-one-out approach where each subject's time series (per voxel) is
    correlated with the average of the remaining subjects' time series. The
    --input should be two or more 4-dimensional NIfTI (.nii or .nii.gz) files, one
    for each subject. Alternatively, a wildcard can be used to indicate multiple
    files (e.g., *.nii.gz). The --output should be a single filename indicating
    where the results of the ISC analysis should be saved. If N subjects are input
    into the ISC analysis, the resulting output file will have N samples (one ISC
    value for each left-out subject). Typically a 3-dimensional NIfTI file should
    be supplied to the --mask argument so as to restrict the analysis to voxels of
    interest (e.g., the brain, gray matter). The mask file will be converted to a
    Boolean array and should have 1s for voxels of interest and 0s elsewhere. All
    input files (and the mask) must be spatially normalized to standard space
    (e.g., MNI space) prior to ISC analysis. The --zscore argument indicates that
    response time series should be z-scored (per voxel) prior to ISC analysis;
    this may be important when computing the average time series for N–1 subjects.
    The --summarize option can be used to computer either the mean or median ISC
    value across left-out subjects after completing the ISC analysis (in which case
    the output file will only have one sample). If mean ISC values are requested,
    ISC values are Fisher Z-transformed, the mean is computed, and then the mean
    is inverse Fisher Z-transformed. This program requires an installation of
    Python 3 with the NumPy/SciPy and NiBabel modules. The implementation is
    based on the BrainIAK (https://brainiak.org) implementation, but does not
    require a BrainIAK installation. Note that this software is not written for
    speed or memory-efficiency, but for readability/transparency.

    Example usage:
        isc_cli.py --input s1.nii.gz s2.nii.gz s3.nii.gz \\
        --output isc.nii.gz --mask mask.nii.gz --zscore

        isc_cli.py --input s*.nii.gz --output isc.nii.gz \\
        --mask mask.nii.gz --zscore

        isc_cli.py --input s*.nii.gz --output mean_isc.nii.gz \\
        --mask mask.nii.gz --zscore --summarize mean

    References:
    Nastase, S. A., Gazzola, V., Hasson, U., Keysers, C. (in preparation).
    Measuring shared responses across subjects using intersubject correlation.

    Author: Samuel A. Nastase, 2019
        """), formatter_class=argparse.RawDescriptionHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required_arguments')
    required.add_argument("-i", "--inputs", nargs='+', required=True,
                          help=("NIfTI input files on which to compute ISCs"))
    required.add_argument("-o", "--output", type=str, required=True,
                          help=("NIfTI output filename for ISC values"))
    optional.add_argument("-m", "--mask", type=str,
                        help=("NIfTI mask file for masking input data"))
    optional.add_argument("-z", "--zscore", action='store_true',
                        dest='zscore_data',
                        help=("Z-score time series for each voxel in input"))
    optional.add_argument("-s", "--summarize", type=str,
                        choices=['mean', 'median'],
                        help=("summarize results across participants "
                              "using either 'mean' or 'median'"))
    optional.add_argument("-v", "--verbosity", type=int, choices=[1, 2, 3, 4, 5],
                        default=3, help=("increase output verbosity via "
                                         "Python's logging module"))
    parser.add_argument('--version', action='version', version='isc_cli.py 1.0.0')
    parser._action_groups.append(optional)
    args = parser.parse_args(args)
    
    return args


# Function to load NIfTI mask and convert to boolean
def load_mask(mask_arg):

    # Load mask from file
    mask = nib.load(mask_arg).get_fdata().astype(bool)

    # Get indices of voxels in mask
    mask_indices = np.where(mask)

    return mask, mask_indices


# Function for loading (and organizing) data from NIfTI files
def load_data(inputs_arg, mask=None):

    # Convert input argument string to filenames
    input_fns = [fn for fns in [glob(fn) for fn in inputs_arg]
                     for fn in fns]
    data = []
    affine, data_shape = None, None
    for input_fn in input_fns:

        # Load in the NIfTI image using NiBabel and check shapes
        input_img = nib.load(input_fn)
        if not data_shape:
            data_shape = input_img.shape
            shape_fn = input_fn
        if input_img.ndim != 4:
            raise ValueError("input files should be 4-dimensional "
                             "(three spatial dimensions plus time)")
        if input_img.shape != data_shape:
            raise ValueError("input files have mismatching shape: "
                             "file '{0}' with shape {1} does not "
                             "match file '{2}' with shape "
                             "{3}".format(input_fn,
                                          input_img.shape,
                                          shape_fn,
                                          data_shape))
        logger.debug("input file '{0}' NIfTI image is "
                     "shape {1}".format(input_fn, data_shape))

        # Save the affine and header from the first image
        if affine is None:
            affine, header = input_img.affine, input_img.header
            logger.debug("using affine and header from "
                         "file '{0}'".format(input_fn))

        # Get data from image and apply mask (if provided)
        input_data = input_img.get_fdata()
        if isinstance(mask, np.ndarray):
            input_data = input_data[mask]
        else:
            input_data = input_data.reshape((
                np.product(input_data.shape[:3]),
                input_data.shape[3]))
        data.append(input_data.T)
        logger.info("finished loading data from "
                    "'{0}'".format(input_fn))

    # Stack input data
    data = np.stack(data, axis=2)

    return data, affine, header


# Function to compute leave-one-out ISCs on input data
def compute_iscs(data):

    # Get shape of data
    n_TRs, n_voxels, n_subjects = data.shape

    # Check if only two subjects
    if n_subjects == 2:
        logger.warning("only two subjects provided! simply "
                       "computing ISC between them")

    # Loop over each voxel or ROI
    voxel_iscs = []
    for v in np.arange(data.shape[1]):
        voxel_data = data[:, v, :].T

        # Compute Pearson correlations between voxel time series
        if n_subjects == 2:
            iscs = pearsonr(voxel_data[0, :], voxel_data[1, :])[0]
        else:
            iscs = np.array([pearsonr(subject,
                                      np.nanmean(np.delete(
                                                  voxel_data,
                                                  s, axis=0),
                                                 axis=0))[0]
                             for s, subject in enumerate(voxel_data)])
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)

    return iscs


# Function to optionally summarize ISCs
def summarize_iscs(iscs, summary_statistic):

    # Compute mean (with Fisher Z transformation)
    if summary_statistic == 'mean':
        statistic = np.tanh(np.nanmean(np.arctanh(iscs),
                                       axis=0))[np.newaxis, :]
        logger.info("computing mean of ISCs (with "
                    "Fisher Z transformation)")

    # Compute median
    elif summary_statistic == 'median':
        statistic = np.nanmedian(iscs, axis=0)[np.newaxis, :]
        logger.info("computing median of ISCs")

    return statistic


# Function to transform data back into NIfTI image and save
def save_data(iscs, affine, header, output_fn, mask_indices=None):

    # Output ISCs image shape
    i, j, k = header.get_data_shape()[:3]
    output_shape = (i, j, k, iscs.shape[0])

    # Reshape masked data
    if mask_indices:
        output_iscs = np.zeros(output_shape)
        output_iscs[mask_indices] = iscs.T
    else:
        output_iscs = iscs.T.reshape(output_shape)

    # Construct output NIfTI image
    output_img = nib.Nifti1Image(output_iscs, affine)

    # Save the NIfTI image according to output filename
    nib.save(output_img, output_fn)


# Function to execute the above code
def main(args):

    # Get arguments
    args = parse_arguments(args)

    # Set up logger according to verbosity level
    logging.basicConfig(level=abs(6 - args.verbosity) * 10)
    logger.info("verbosity set to Python logging level '{0}'".format(
        logging.getLevelName(logger.getEffectiveLevel())))

    # Get optional mask
    if args.mask:
        mask, mask_indices = load_mask(args.mask)
    else:
        mask, mask_indices = None
        logger.warning("no mask provided! are you sure you want "
                        "to compute ISCs for all voxels in image?")

    # Load data
    data, affine, header = load_data(args.inputs, mask=mask)

    # Optionally z-score data
    if args.zscore_data:
        data = zscore(data)
        logging.info("z-scored input data prior to computing ISCs")

    # Compute ISCs
    iscs = compute_iscs(data)

    # Optionally apply summary statistic
    if args.summarize and iscs.shape[0] > 1:
        iscs = summarize_iscs(iscs,
                              summary_statistic=args.summarize)

    # Save output ISCs to file
    save_data(iscs, affine, header, args.output,
              mask_indices=mask_indices)


# Name guard so we can load these functions elsewhere
# without actually trying to run everything
if __name__ == '__main__':
    main(sys.argv[1:])
