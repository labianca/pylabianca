from . import (io, utils, decoding, localize, selectivity, spikes,
               spike_distance, stats, viz)

from .viz import plot_spike_rate, plot_raster, plot_spikes
from .spikes import SpikeEpochs, Spikes
from .selectivity import depth_of_selectivity, explained_variance