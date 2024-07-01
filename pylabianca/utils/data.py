import os
import os.path as op
import numpy as np


def get_data_path():
    home_dir = os.path.expanduser('~')
    data_dir = 'pylabianca_data'
    full_data_dir = op.join(home_dir, data_dir)
    has_data_dir = op.exists(full_data_dir)

    if not has_data_dir:
        os.mkdir(full_data_dir)

    return full_data_dir


def get_fieldtrip_data():
    import pooch

    data_path = get_data_path()
    ft_url = ('https://download.fieldtriptoolbox.org/tutorial/spike/p029_'
              'sort_final_01.nex')
    known_hash = ('4ae4ed2a9613cde884b62d8c5713c418cff5f4a57c8968a3886'
                  'db1e9991a81c9')
    fname = pooch.retrieve(
        url=ft_url, known_hash=known_hash,
        fname='p029_sort_final_01.nex', path=data_path
    )
    return fname


def get_zeta_example_data():
    import pooch

    data_path = get_data_path()
    github_url = 'https://github.com/JorritMontijn/zetapy/raw/master/zetapy/ExampleDataZetaTest.mat'


    known_hash = ('af93a1887e8afcdfe9a6d212dc6c928fede46d31b184d91288f4f862'
                  'dfddc59f')
    fname = pooch.retrieve(
        url=github_url, known_hash=known_hash,
        fname='zeta_example_data.mat', path=data_path
    )
    return fname


def get_test_data_link():
    dropbox_lnk = ('https://www.dropbox.com/scl/fo/757tf3ujqga3sa2qocm4l/h?'
                   'rlkey=mlz44bcqtg4ds3gsc29b2k62x&dl=1')
    return dropbox_lnk


def download_test_data():
    # check if test data exist
    data_dir = get_data_path()
    check_files = [
        'ft_spk_epoched.mat', 'monkey_stim.csv',
        'p029_sort_final_01_events.mat',
        op.join('test_osort_data', 'sub-U04_switchorder',
                'CSCA130_mm_format.mat'),
        op.join('test_neuralynx', 'sub-U06_ses-screening_set-U6d_run-01_ieeg',
                'CSC129.ncs')
    ]

    if all([op.isfile(op.join(data_dir, f)) for f in check_files]):
        return

    import pooch
    import zipfile

    # set up paths
    fname = 'temp_file.zip'
    download_link = get_test_data_link()

    # download the file
    hash = None
    pooch.retrieve(url=download_link, known_hash=hash,
                   path=data_dir, fname=fname)

    # unzip and extract
    # TODO - optionally extract only the missing files
    destination = op.join(data_dir, fname)
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()

    # remove the zipfile
    os.remove(destination)


def create_random_spikes(n_cells=4, n_trials=25, n_spikes=(10, 21),
                         **args):
    '''Create random spike data. Mostly useful for testing.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_trials : int
        Number of trials. If ``None`` or 0 then Spikes object is returned.
    n_spikes : int | tuple
        Number of spikes. If tuple then the first element is the minimum
        number of spikes and the second element is the maximum number of
        spikes.
    args : dict
        Additional arguments are passed to the Spikes / SpikeEpochs object.

    Returns
    -------
    spikes : Spikes | SpikeEpochs
        Spike data object.
    '''
    from ..spikes import SpikeEpochs, Spikes

    tmin, tmax = -0.5, 1.5
    tlen = tmax - tmin
    constant_n_spikes = isinstance(n_spikes, int)
    if constant_n_spikes:
        n_spk = n_spikes

    return_epochs = isinstance(n_trials, int) and n_trials > 0
    if not return_epochs:
        n_trials = 1
        tmin = 0
        tmax = 1e6

    times = list()
    trials = list()
    for _ in range(n_cells):
        this_tri = list()
        this_tim = list()
        for tri_idx in range(n_trials):
            if not constant_n_spikes:
                n_spk = np.random.randint(*n_spikes)

            if return_epochs:
                tms = np.random.rand(n_spk) * tlen + tmin
                this_tri.append(np.ones(n_spk, dtype=int) * tri_idx)
            else:
                tms = np.random.randint(tmin, tmax, size=n_spk)
            tms = np.sort(tms)
            this_tim.append(tms)

        this_tim = np.concatenate(this_tim)
        times.append(this_tim)

        if return_epochs:
            this_tri = np.concatenate(this_tri)
            trials.append(this_tri)

    if return_epochs:
        return SpikeEpochs(times, trials, time_limits=(tmin, tmax), **args)
    else:
        if 'sfreq' not in args:
            args['sfreq'] = 10_000

        return Spikes(times, **args)
