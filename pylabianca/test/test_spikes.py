import numpy as np
import pylabianca as pln


def create_fake_spikes():
    times = [[-0.3, -0.1, 0.025, 0.11, 0.22, 0.25, 0.4,
            -0.08, 0.12, 0.14, 0.19, 0.23, 0.32],
            [-0.22, -0.13, -0.03, 0.08, 0.16, 0.33, -0.2,
            -0.08, 0.035, 0.148, 0.32]]
    trials = [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    spk = pln.SpikeEpochs(times, trials)
    return spk


def test_crop():
    spk = create_fake_spikes()
    spk.crop(tmin=0.1)

    assert (spk.time[0] == np.array(
        [0.11, 0.22, 0.25, 0.4, 0.12, 0.14, 0.19, 0.23, 0.32])).all()
    assert (spk.time[1] == np.array([0.16, 0.33, 0.148, 0.32])).all()
    assert (spk.trial[0] == np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])).all()
    assert (spk.trial[1] == np.array([0, 0, 1, 1])).all()


def test_repr():
    spk = create_fake_spikes()
    shouldbe = '<SpikeEpochs, 2 epochs, 2 cells, 12.0 spikes/cell on average>'
    assert str(spk) == shouldbe


def test_pick_cells():
    spk = create_fake_spikes()
    assert len(spk.time) == 2
    spk.picks_cells('cell001')
    assert len(spk.time) == 1
    assert spk.time[0][0] == -0.22


def test_to_raw():
    times = [[-0.3, -0.28, -0.26, 0.15, 0.18, 0.2],
             [-0.045, 0.023, -0.1, 0.13]]
    trials = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1]]
    spk = pln.SpikeEpochs(times, trials)

    spk_tm, spk_raw = pln.spikes._spikes_to_raw(spk, sfreq=10)
    print(spk_raw)
    good_tms = np.arange(-0.3, 0.21, step=0.1)
    good_raw = np.array(
        [[[3, 0, 0, 0, 1, 2], [0, 0, 0, 2, 0, 0]],
         [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0]]]
    )
    assert (spk_tm == good_tms).all()
    assert (spk_raw == good_raw).all()