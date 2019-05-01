import tempfile
import subprocess
import pytest
from glob import glob
from os.path import join
import numpy as np
import nibabel as nib
from test_isc_standalone import (simulated_timeseries,
                                 correlated_timeseries)
from isc_cli import (parse_arguments, load_mask,
                     load_data, compute_iscs,
                     summarize_iscs, save_data,
                     main)


def simulate_nifti(i, j, k, n_TRs=None, noise=None,
                   mask=False, random_seed=None):

    # Allow for fixed random seed
    prng = np.random.RandomState(random_seed)

    # Create random normal data
    if n_TRs:
        data = prng.randn(i, j, k, n_TRs)
    else:
        data = prng.randn(i, j, k)

    # Optionally add noise
    if noise:
        data +- prng.randn(*data.shape) * noise

    # Optionally binarize image into 1s and 0s mask
    if mask:
        data = np.where(data > 0, 1, 0)

    # Create dummy affine matrix
    affine = np.eye(4)

    # Convert array into NIfTI image
    data_img = nib.Nifti1Image(data, affine)

    return data_img


def test_parse_arguments():

    # Check that help renders
    with pytest.raises(SystemExit):
        args = parse_arguments(['--help'])

    # Check default arguments
    args = parse_arguments(['--input', 's1.nii.gz', 's2.nii.gz',
                            's3.nii.gz', '--output', 'isc.nii.gz'])
    assert args.input == ['s1.nii.gz', 's2.nii.gz', 's3.nii.gz']
    assert args.output == 'isc.nii.gz'
    assert args.mask == None
    assert args.summarize == None
    assert args.verbosity == 3
    assert args.apply_zscore == False
    assert args.apply_fisherz == False

    # Check optional arguments
    args = parse_arguments(['--input', 's1.nii.gz', 's2.nii.gz',
                            's3.nii.gz', '--output', 'isc.nii.gz',
                            '--mask', 'mask.nii.gz', '--zscore',
                            '--fisherz', '--summarize', 'mean',
                            '--verbosity', '2'])
    assert args.mask == 'mask.nii.gz'
    assert args.summarize == 'mean'
    assert args.verbosity == 2
    assert args.apply_zscore == True
    assert args.apply_fisherz == True

    # Check abbreviated arguments
    args = parse_arguments(['-i', 's1.nii.gz', 's2.nii.gz',
                            's3.nii.gz', '-o', 'isc.nii.gz',
                            '-m', 'mask.nii.gz', '-z',
                            '-f', '-s', 'mean',
                            '-v', '2'])
    assert args.mask == 'mask.nii.gz'
    assert args.summarize == 'mean'
    assert args.verbosity == 2
    assert args.apply_zscore == True
    assert args.apply_fisherz == True


def test_load_mask():

    i, j, k = 10, 10, 10

    # Create dummy mask image
    mask_img = simulate_nifti(i, j, k, mask=True)

    # Create a temporary directory to load files froom
    with tempfile.TemporaryDirectory() as temp_dir:
        mask_img.to_filename(join(temp_dir, 'tmp_mask.nii.gz'))

        mask, mask_indices = load_mask(join(temp_dir, 'tmp_mask.nii.gz'))

    assert np.array_equal(mask_img.get_fdata(), mask)


def test_load_data():

    i, j, k, n_TRs = 10, 10, 10, 300
    n_subjects = 3

    # Create a temporary directory to load files from
    with tempfile.TemporaryDirectory() as temp_dir:

        # Create example datasets
        data_imgs = []
        for subject in np.arange(n_subjects):
            data_imgs.append(simulate_nifti(i, j, k, n_TRs, noise=1,
                                            random_seed=subject))

        for subject, data_img in enumerate(data_imgs):
            data_img.to_filename(join(temp_dir,
                                      'tmp_s{0}.nii.gz'.format(
                                          subject)))
        data, affine, header = load_data(glob(
            join(temp_dir, 'tmp_s*.nii.gz')))

        # Check for exception if we only get one input file
        with pytest.raises(ValueError):
            data, affine, header = load_data(glob(
                join(temp_dir, 'tmp_s1.nii.gz')))

        # Create datasets with only 3 dimensions
        data_imgs = []
        for subject in np.arange(n_subjects):
            data_imgs.append(simulate_nifti(i, j, k, noise=1,
                                            random_seed=subject))

        for subject, data_img in enumerate(data_imgs):
            data_img.to_filename(join(temp_dir,
                                      'tmp_s{0}.nii.gz'.format(
                                          subject)))

        with pytest.raises(ValueError):
            data, affine, header = load_data(glob(
                        join(temp_dir, 'tmp_s*.nii.gz')))

        # Create mismatching sized datasets
        data_imgs = []
        for subject in np.arange(n_subjects):
            data_imgs.append(simulate_nifti(i, j, k, n_TRs + subject,
                                            noise=1, random_seed=subject))

        for subject, data_img in enumerate(data_imgs):
            data_img.to_filename(join(temp_dir,
                                      'tmp_s{0}.nii.gz'.format(
                                          subject)))

        with pytest.raises(ValueError):
            data, affine, header = load_data(glob(
                        join(temp_dir, 'tmp_s*.nii.gz')))

    assert np.array_equal(affine, np.eye(4))
    assert data.shape == (n_TRs, np.product((i, j, k)),
                          n_subjects)


def test_compute_iscs():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    # Check basic computation and output shape
    iscs = compute_iscs(data)
    assert iscs.shape == (n_subjects, n_voxels)

    # Just two subjects
    iscs = compute_iscs(data[..., :2])
    assert iscs.shape == (1, n_voxels)

    # Correlated time series
    data = correlated_timeseries(20, 60, noise=0,
                                 random_state=42)
    iscs = compute_iscs(data)
    assert np.allclose(iscs[:, :2], 1., rtol=1e-05)
    assert np.all(iscs[:, -1] < 1.)

    iscs = compute_iscs(data)
    assert np.allclose(iscs[:, :2], 1., rtol=1e-05)
    assert np.all(iscs[:, -1] < 1.)


def test_summarize_iscs():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    # Check shape of mean and median outputs
    iscs = compute_iscs(data)
    mean_iscs = summarize_iscs(iscs, 'mean')
    assert mean_iscs.shape == (1, n_voxels)

    median_iscs = summarize_iscs(iscs, 'median')
    assert median_iscs.shape == (1, n_voxels)

    # Dummy ISCs matrix with known mean/median
    central = .5
    iscs = np.array([[central-.25, central-.25],
                     [central, central],
                     [central, central],
                     [central+.25, central+.25]])
    assert np.allclose(summarize_iscs(iscs, 'mean'),
                        np.array([[central, central]]),
                        atol=.1)
    assert np.array_equal(summarize_iscs(iscs, 'median'),
                          np.array([[central, central]]))


def test_save_data():

    i, j, k = 10, 10, 10
    n_subjects = 3
    iscs = np.random.randn(n_subjects, np.product((i, j, k)))
    affine = np.eye(4)
    header = nib.Nifti1Image(np.zeros((i, j, k)), affine).header
    mask_indices = np.where(np.ones((i, j, k)))

    # Create a temporary directory to save files
    with tempfile.TemporaryDirectory() as temp_dir:
        save_data(iscs, affine, header,
                  join(temp_dir, 'test_iscs.nii.gz'),
                  mask_indices=mask_indices)

        data = nib.load(join(temp_dir,
                             'test_iscs.nii.gz')).get_data()
        data = data.reshape(np.product((i, j, k)), n_subjects).T

    assert np.array_equal(data, iscs)


def test_main():

    # Create mini input data and mask
    i, j, k, n_TRs = 10, 10, 10, 300
    n_subjects = 3

    data_imgs = []
    for subject in np.arange(n_subjects):
        data_imgs.append(simulate_nifti(i, j, k, n_TRs, noise=1,
                                        random_seed=subject))

    mask_img = simulate_nifti(i, j, k, mask=True)

    # Create a temporary directory to load files froom
    with tempfile.TemporaryDirectory() as temp_dir:

        for subject, data_img in enumerate(data_imgs):
            data_img.to_filename(join(temp_dir,
                                      'tmp_s{0}.nii.gz'.format(
                                          subject)))

        mask_img.to_filename(join(temp_dir, 'tmp_mask.nii.gz'))

        main(['--input', join(temp_dir, 'tmp_s*.nii.gz'),
              '--output', join(temp_dir, 'tmp_iscs.nii.gz'),
              '--mask', join(temp_dir, 'tmp_mask.nii.gz'),
              '--zscore', '--fisherz', '--summarize', 'mean',
              '--verbosity', '4'])


if __name__ == '__main__':
    test_parse_arguments()
    test_load_mask()
    test_load_data()
    test_compute_iscs()
    test_summarize_iscs()
    test_save_data()
    test_main()
