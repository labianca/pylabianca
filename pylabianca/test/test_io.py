import os.path as op
import numpy as np
import pytest

import pylabianca as pln
from pylabianca.utils import download_test_data, get_data_path


download_test_data()
data_dir = get_data_path()


def test_read_osort(tmp_path):
    osort_dir = op.join(data_dir, r'test_osort_data\sub-U04_switchorder')

    # read without waveforms
    spk = pln.io.read_osort(osort_dir, waveform=False)

    assert spk.cellinfo.shape[0] == len(spk.cell_names)
    assert len(spk.cell_names) == len(spk.timestamps)
    assert spk.waveform is None

    uni_ch = np.unique(spk.cellinfo.channel)
    assert len(uni_ch) == 3

    # now read with waveforms
    spk = pln.io.read_osort(osort_dir, waveform=True)

    assert len(spk.waveform) == len(spk.cell_names)
    for cell_idx in range(spk.n_units()):
        assert spk.waveform[cell_idx].shape[0] == len(spk.timestamps[cell_idx])

    # make sure we get error when format is misspecified
    msg = 'Could not find the "assignedNegative" field'
    with pytest.raises(ValueError, match=msg):
        pln.io.read_osort(osort_dir, format='standard')

    # and an error when the format is not recognized
    match_str = 'Unrecognized format "lieber_biber"'
    with pytest.raises(ValueError, match=match_str):
        pln.io.read_osort(osort_dir, format='lieber_biber')

    # TODO: specify channel picks
    # and compare using cellinfo etc.

    # SAVE to matlab
    # --------------
    fname = 'test.mat'
    spk.to_matlab(op.join(tmp_path, fname))

    # read it back and compare
    spk2 = pln.io.read_osort(op.join(tmp_path, fname))

    n_units = spk.n_units()
    assert len(spk) == len(spk2)
    assert all(spk.cell_names == spk2.cell_names)
    assert all([(spk.timestamps[ix] == spk2.timestamps[ix]).all()
                for ix in range(n_units)])
    assert all([(spk.waveform[ix] == spk2.waveform[ix]).all()
                for ix in range(n_units)])

    # (NOT IMPLEMENTED) save in standard format
    # fname = 'test_standard.mat'
    # spk.to_matlab(op.join(tmp_path, fname), format='osort_standard')

