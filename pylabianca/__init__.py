from . import io, utils, co_spikes, spikes, spike_distance, viz
from .viz import plot_spike_rate
from .io import read_gammbur, read_spikes
from .spikes import (SpikeEpochs, Spikes)
from .selectivity import (depth_of_selectivity, explained_variance)
from .co_spikes import shuffled_spike_xcorr