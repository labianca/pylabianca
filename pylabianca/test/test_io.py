import os.path as op
import numpy as np
import pandas as pd
import pytest

import pylabianca as pln
from pylabianca.utils import download_test_data, get_data_path
from pylabianca.testing import random_spikes


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
    spk = random_spikes()
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

    # one unit has exactly one spike
    # (this used to cause scipy.io.loadmat to squeeze out the value and
    #  make the element non-nparray)
    unit_idx = 0
    n_spikes_first_unit = len(spk.time[unit_idx])
    select_spike_idx = np.random.randint(0, n_spikes_first_unit)
    spk_one_spike = spk.copy()
    spk_one_spike.time[0] = spk_one_spike.time[0][select_spike_idx]
    spk_one_spike.trial[0] = spk_one_spike.trial[0][select_spike_idx]
    spk_one_spike.waveform = None
    spk_one_spike.waveform_time = None
    io_roundtrip(spk_one_spike, filepath, kind='trials')

    # only one unit in the file
    spk_one_unit = random_spikes(n_cells=1)
    io_roundtrip(spk_one_unit, filepath, kind='trials')

    # io roundtrip for Spikes
    filepath = op.join(tmp_path, 'spikeRaw.mat')
    spk_raw = random_spikes(
        n_cells=3, n_trials=0, n_spikes=(23, 55))
    spk_raw.cellinfo = cellinfo.iloc[:-1, :]
    io_roundtrip(spk_raw, filepath, kind='raw')


def test_from_spiketools_times_roundtrip():
    spike_times = np.array([0.1, 0.25, 0.7])

    spk = pln.io.from_spiketools(spike_times, kind='times')

    assert isinstance(spk, pln.SpikeEpochs)
    assert spk.n_trials == 1
    assert spk.n_units() == 1
    np.testing.assert_array_equal(spk.time[0], spike_times)
    np.testing.assert_array_equal(
        spk.trial[0], np.zeros(len(spike_times), dtype=int))

    roundtrip = pln.io.to_spiketools(spk)
    assert len(roundtrip) == 1
    np.testing.assert_array_equal(roundtrip[0], spike_times)


def test_from_spiketools_trials_roundtrip():
    trial_spikes = [
        np.array([0.1, 0.2]),
        np.array([]),
        np.array([0.4]),
    ]

    spk = pln.io.from_spiketools(trial_spikes, kind='trials')
    roundtrip = pln.io.to_spiketools(spk)

    assert spk.n_trials == len(trial_spikes)
    assert len(roundtrip) == len(trial_spikes)
    for actual, expected in zip(roundtrip, trial_spikes):
        np.testing.assert_array_equal(actual, expected)


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


def _minimal_ncs_records(n_records=2):
    from pylabianca.neuralynx_io import NCS_RECORD

    records = np.zeros(n_records, dtype=NCS_RECORD)
    records['TimeStamp'] = np.arange(n_records, dtype=np.uint64) * 256000
    records['ChannelNumber'] = 1
    records['SampleFreq'] = 2000
    records['NumValidSamples'] = 512
    records['Samples'] = np.arange(
        n_records * 512, dtype=np.int16).reshape(n_records, 512)

    return records


def _minimal_ncs_header(old_preamble=(), cheetah_rev='5.6.3'):
    from pylabianca.neuralynx_io import HEADER_LENGTH

    lines = [
        '######## Neuralynx Data File Header',
        *old_preamble,
        '-FileType CSC',
        '-FileVersion 3.3.0',
        '-RecordSize 1044',
        '-CheetahRev ' + cheetah_rev,
        '-NLX_Base_Class_Name CSC17',
        '-NLX_Base_Class_Type CscAcqEnt',
        '-SamplingFrequency 2000',
        '-ADBitVolts 0.000000061037020770982053',
        '-ADMaxValue 32767',
    ]
    raw_header = '\r\n'.join(lines).encode('ascii')
    return raw_header + b'\0' * (HEADER_LENGTH - len(raw_header))


def _old_ncs_header_case(file_name_line, time_opened_line, cheetah_rev,
                         time_closed_line=None, case_id=None):
    preamble = [file_name_line, time_opened_line]
    expected = {
        'FileName': file_name_line.split('File Name', 1)[1].lstrip(': '),
        'TimeOpened': 'Time Opened',
        'FileType': 'CSC',
        'CheetahRev': cheetah_rev,
    }
    if time_closed_line is not None:
        preamble.append(time_closed_line)
        expected['TimeClosed'] = 'Time Closed'

    return pytest.param(
        {'preamble': preamble, 'expected': expected},
        id=case_id)


def test_neuralynx_estimate_record_count(tmp_path):
    from pylabianca.neuralynx_io import (
        HEADER_LENGTH, NCS_RECORD, estimate_record_count)

    n_records = 3
    ncs_file = tmp_path / 'test_record_count.ncs'
    ncs_file.write_bytes(
        b'\0' * HEADER_LENGTH
        + b'\0' * (n_records * NCS_RECORD.itemsize))

    assert estimate_record_count(ncs_file, NCS_RECORD) == n_records

    ncs_file.write_bytes(ncs_file.read_bytes() + b'x')
    expected = n_records + 1 / NCS_RECORD.itemsize
    with pytest.warns(UserWarning, match='not divisible by record size'):
        assert estimate_record_count(ncs_file, NCS_RECORD) == expected

    too_small_file = tmp_path / 'too_small.ncs'
    too_small_file.write_bytes(b'\0' * (HEADER_LENGTH - 1))
    with pytest.raises(ValueError, match='Too small to be a valid .ncs file'):
        estimate_record_count(too_small_file, NCS_RECORD)


@pytest.mark.parametrize(
    'case',
    [
        _old_ncs_header_case(
            '## File Name C:\\CheetahData\\2013-08-18_09-06-16\\CSC17.ncs',
            '## Time Opened (m/d/y): 8/18/2013 (h:m:s.ms) 9:6:36.401',
            '5.6.3',
            time_closed_line=(
                '## Time Closed (m/d/y): 8/18/2013 '
                '(h:m:s.ms) 10:26:2.464'),
            case_id='cheetah-5-with-time-closed'),
        _old_ncs_header_case(
            '## File Name: D:\\Cheetah_Data\\2003-2-26_13-9-56\\CSC4.Ncs',
            '## Time Opened: (m/d/y): 2/26/2003 At Time: 13:9:58.250',
            '3.0.6',
            case_id='cheetah-3-without-time-closed'),
    ]
)
def test_neuralynx_old_format_ncs_header(tmp_path, case):
    from pylabianca.neuralynx_io import load_ncs, write_ncs

    expected = case['expected']
    raw_header = _minimal_ncs_header(
        case['preamble'], cheetah_rev=expected['CheetahRev'])
    records = _minimal_ncs_records()
    ncs_file = tmp_path / 'old_format.ncs'
    write_ncs(ncs_file, records, raw_header)

    data = load_ncs(ncs_file, load_time=False, rescale_data=False)

    for key, value in expected.items():
        if key.startswith('Time'):
            assert data['header'][key].startswith(value)
        else:
            assert data['header'][key] == value
    assert ('TimeClosed' in data['header']) == ('TimeClosed' in expected)
    assert data['data'].shape == (records.size * 512,)
    np.testing.assert_array_equal(data['data'][:512], records[0]['Samples'])


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

    # Remove the ADBitVolts line and pad to correct length
    header_str = raw_header.decode('ascii', errors='ignore')
    header_lines = [line for line in header_str.splitlines()
                    if not line.strip().startswith("-ADBitVolts")]
    stripped_header = '\r\n'.join(header_lines).encode('ascii')
    stripped_header = (stripped_header[:HEADER_LENGTH]
                       + b'\0' * (HEADER_LENGTH - len(stripped_header)))

    new_fname = fname.replace('.ncs', '_no_scaling_info.ncs')
    output_file = op.join(tmp_path, new_fname)
    write_ncs(output_file, records[:10], stripped_header)

    data = load_ncs(output_file, load_time=False, rescale_data=False)
    assert data['data'].dtype == np.int16

    with pytest.warns(UserWarning, match='Unable to rescale data'):
        data = load_ncs(output_file, load_time=False)

    assert data['data'].dtype == np.int16
    assert (data['data'][:512] == records[0]['Samples']).all()


def test_add_region_from_channel_ranges():
    # create random spikes
    spk = random_spikes(
        n_cells=10, cell_names=list('ABCDEFGHIJ'))

    # create cellinfo with channel numbers
    ch_num = np.arange(1, 11)
    cellinfo = pd.DataFrame(data={'channel': ch_num})
    spk.cellinfo = cellinfo

    # create a table with anatomy info
    region_info = pd.DataFrame(
        data={'channel start': [1, 5, 10], 'channel end': [4, 9, 10],
              'region': ['AMY', 'HIP', 'ACC']})

    # add anatomy info to spk
    pln.io.add_region_from_channel_ranges(
        spk, region_info, source_column='region', target_column='anat')

    assert (spk.cellinfo.anat == ['AMY'] * 4 + ['HIP'] * 5 + ['ACC']).all()

    # test the same, but now with some channel ranges missing
    spk.cellinfo = spk.cellinfo.drop(columns='anat')
    region_info_missing = region_info.drop(index=1)

    pln.io.add_region_from_channel_ranges(
        spk, region_info_missing, source_column='region', target_column='anat')

    correct = np.array(['AMY'] * 4 + [np.nan] * 5 + ['ACC'], dtype=object)
    not_nan = ~pd.isna(correct)
    assert (spk.cellinfo.anat[not_nan] == correct[not_nan]).all()
    assert (pd.isna(spk.cellinfo.anat[~not_nan])).all()
