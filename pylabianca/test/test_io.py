import os.path as op
import numpy as np
import pandas as pd
import pytest

import pylabianca as pln
from pylabianca.utils import download_test_data, get_data_path


download_test_data()
data_dir = get_data_path()


def test_read_osort(tmp_path):
    osort_dir = op.join(
        data_dir, 'test_osort_data', 'sub-U04_switchorder')

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

def test_read_events_neuralynx():
    lynx_dir = op.join(
        data_dir, 'test_neuralynx',
        'sub-U06_ses-screening_set-U6d_run-01_ieeg'
    )
    events_df = pln.io.read_events_neuralynx(lynx_dir)

    # one experiment start trigger
    assert (events_df.trigger == 61).sum() == 1
    assert (events_df.trigger == 1).sum() == (6 * 63)

    # read mne events format
    events = pln.io.read_events_neuralynx(lynx_dir, format='mne')
    assert isinstance(events, np.ndarray)
    assert events.ndim == 2
    assert events.shape[1] == 3

    events_df_actual = events_df.query('trigger >= 0')
    assert (events_df_actual.timestamp.values == events[:, 0]).all()

    events_df2 = pln.io.read_events_neuralynx(
        lynx_dir, first_timestamp_from=None)
    assert (events_df_actual.timestamp.values
            == events_df2.timestamp.values).all()

    with pytest.raises(ValueError, match='Unknown format'):
        pln.io.read_events_neuralynx(lynx_dir, format='lieber_biber')


def test_read_events_neuralynx_include_zero_triggers_mne():
    lynx_dir = op.join(
        data_dir, 'test_neuralynx',
        'sub-U06_ses-screening_set-U6d_run-01_ieeg'
    )

    # Read events with zero triggers included
    events_include_zero = pln.io.read_events_neuralynx(lynx_dir, format='mne',
                                                       ignore_zero=False)

    # Verify that triggers include zero
    assert (events_include_zero[:, -1] == 0).any()

    # Compare with events where zero triggers are ignored
    events_ignore_zero = pln.io.read_events_neuralynx(lynx_dir, format='mne',
                                                      ignore_zero=True)
    assert len(events_include_zero) > len(events_ignore_zero)


def make_sure_identical(spk, spk2):
    n_units = spk.n_units()
    assert (spk.cell_names == spk2.cell_names).all()
    assert n_units == spk2.n_units()

    is_epochs = isinstance(spk, pln.SpikeEpochs)
    is_epochs2 = isinstance(spk2, pln.SpikeEpochs)

    if is_epochs:
        assert is_epochs2
    else:
        is_raw = isinstance(spk, pln.Spikes)
        is_raw2 = isinstance(spk2, pln.Spikes)
        assert is_raw and is_raw2

    # time_limits are tuple, but are read as array ...
    if is_epochs:
        assert isinstance(spk.time_limits, tuple)
        assert isinstance(spk2.time_limits, tuple)

        assert spk.time_limits == spk2.time_limits

        for cell_idx in range(n_units):
            assert (spk.time[cell_idx] == spk2.time[cell_idx]).all()
            assert (spk.trial[cell_idx] == spk2.trial[cell_idx]).all()
    else:
        for cell_idx in range(n_units):
            assert (spk.timestamps[cell_idx] == spk2.timestamps[cell_idx]).all()

    has_waveform = spk.waveform is not None
    has_waveform2 = spk2.waveform is not None
    assert has_waveform == has_waveform2

    # compare waveform data:
    if has_waveform:
        for cell_idx in range(n_units):
            wave1 = spk.waveform[cell_idx]
            wave2 = spk2.waveform[cell_idx]
            is_none = [x is None for x in [wave1, wave2]]
            assert is_none[0] == is_none[1]
            if not is_none[0]:
                assert (wave1 == wave2).all()

    has_wave_time = spk.waveform_time is not None
    has_wave_time2 = spk2.waveform_time is not None
    assert has_wave_time == has_wave_time2

    if has_wave_time:
        assert (spk.waveform_time == spk2.waveform_time).all()

    if is_epochs:
        has_meta = spk.metadata is not None
        has_meta2 = spk2.metadata is not None
        assert has_meta == has_meta2

        if has_meta:
            assert (spk.metadata == spk2.metadata).all().all()

    has_cellinfo = spk.cellinfo is not None
    has_cellinfo2 = spk2.cellinfo is not None
    assert has_cellinfo == has_cellinfo2

    if has_cellinfo:
        assert (spk.cellinfo == spk2.cellinfo).all().all()


def test_read_write_fieldtrip(tmp_path):
    from string import ascii_lowercase

    def io_roundtrip(spk, filepath, kind='trials'):
        spk.to_fieldtrip(filepath)
        spk2 = pln.io.read_fieldtrip(filepath, kind=kind)
        make_sure_identical(spk, spk2)
        return spk2

    # random spikes
    spk = pln.utils.create_random_spikes()
    n_spk = spk.n_spikes()
    n_tri = spk.n_trials
    n_uni = len(n_spk)

    # create waveforms
    n_smp = 32
    shape = np.sin(np.arange(n_smp) / (n_smp / 4))
    spk.waveform = [
        np.random.normal(scale=0.1, size=(n_spk[idx], n_smp)) + shape
        for idx in range(n_uni)
    ]

    # create cellinfo
    letters = list(ascii_lowercase)
    names = [
        ''.join(
            np.random.choice(letters, size=n_uni).tolist()
        ) for _ in range(n_uni)
    ]
    cellinfo = pd.DataFrame(
        {'cell_name': names,
         'cluster_id': np.random.randint(0, 5000, size=n_uni),
         'area': np.random.choice(['AMY', 'HIP'], size=n_uni)
        }
    )
    spk.cellinfo = cellinfo

    # create metadata
    condition_int = np.random.choice([1, 2, 3], size=n_tri)
    condition_flt = np.random.normal(size=n_tri)
    condition_str = np.random.choice(['A', 'B'], size=n_tri)
    df = pd.DataFrame({'cond': condition_str, 'load': condition_int,
                    'RT': condition_flt})
    spk.metadata = df

    # check io roundtrip
    filepath = op.join(tmp_path, 'spikeTrials.mat')
    io_roundtrip(spk, filepath, kind='trials')

    # when waveform_time is present
    spk.waveform_time = np.linspace(-0.5, 1.5, num=n_smp)
    io_roundtrip(spk, filepath, kind='trials')

    # when one of the cells does not have waveforms
    spk_no_wave = spk.copy()
    spk_no_wave.waveform[2] = None
    io_roundtrip(spk_no_wave, filepath, kind='trials')

    # waveform number of samples does not match
    msg = 'Not all waveforms have the same number of samples'
    spk_no_wave.waveform[1] = spk_no_wave.waveform[1][:, :n_smp - 6]
    spk_no_wave.to_fieldtrip(filepath)

    with pytest.warns(match=msg):
        spk_no_wave2 = pln.io.read_fieldtrip(filepath, kind='trials')
    assert spk_no_wave2.waveform is None

    # no waveforms
    spk_no_wave.waveform = None
    spk_no_wave.waveform_time = None
    io_roundtrip(spk_no_wave, filepath, kind='trials')

    # io roundtrip for Spikes
    filepath = op.join(tmp_path, 'spikeRaw.mat')
    spk_raw = pln.utils.create_random_spikes(
        n_cells=3, n_trials=0, n_spikes=(23, 55))
    spk_raw.cellinfo = cellinfo.iloc[:-1, :]
    io_roundtrip(spk_raw, filepath, kind='raw')


def test_neuralynx_no_records(tmp_path):
    from pylabianca.neuralynx_io import (
        read_raw_header, write_ncs, NCS_RECORD, load_ncs)

    # Read test data file raw header
    path_part = op.join('test_neuralynx',
                        'sub-U06_ses-screening_set-U6d_run-01_ieeg')
    fname = 'CSC129.ncs'
    with open(op.join(data_dir, path_part, fname), 'rb') as fid:
        raw_header = read_raw_header(fid)

    # create and write ncs file containing only the header
    new_fname = fname.replace('.ncs', '_no_data.ncs')
    output_file = op.join(tmp_path, new_fname)
    write_ncs(output_file, np.array([], dtype=NCS_RECORD), raw_header)

    # assert that a warning is raised when reading the file
    msg = 'The file does not contain any data to read'
    with pytest.warns(UserWarning, match=msg):
        data = load_ncs(output_file, load_time=False)
    assert data['data'].shape == (0,)
    assert 'time' not in data

    with pytest.warns(UserWarning, match=msg):
        data = load_ncs(output_file)
    assert data['data'].shape == (0,)
    assert data['time'].shape == (0,)


def test_neuralynx_no_scaling_info(tmp_path):
    from pylabianca.neuralynx_io import (
        read_raw_header, read_records, write_ncs, load_ncs,
        NCS_RECORD, HEADER_LENGTH)

    fname = 'CSC129.ncs'
    path_part = op.join('test_neuralynx',
                        'sub-U06_ses-screening_set-U6d_run-01_ieeg')
    input_file = op.join(data_dir, path_part, fname)
    with open(input_file, 'rb') as fid:
        raw_header = read_raw_header(fid)
        records = read_records(fid, NCS_RECORD)

    # Remove the ADBitVolts line
    header_str = raw_header.decode('ascii', errors='ignore')
    header_lines = [line for line in header_str.splitlines()
                    if not line.strip().startswith("-ADBitVolts")]
    stripped_header = '\r\n'.join(header_lines).encode('ascii')
    stripped_header = stripped_header[:HEADER_LENGTH] + b'\0' * (HEADER_LENGTH - len(stripped_header))

    new_fname = fname.replace('.ncs', '_no_scaling_info.ncs')
    output_file = op.join(tmp_path, new_fname)
    write_ncs(output_file, records[:10], stripped_header)

    data = load_ncs(output_file, load_time=False, rescale_data=False)
    assert data['data'].dtype == np.int16

    with pytest.warns(UserWarning, match='Unable to rescale data'):
        data = load_ncs(output_file, load_time=False)

    assert data['data'].dtype == np.int16
    assert (data['data'][:512] == records[0]['Samples']).all()
