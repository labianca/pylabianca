from . import (io, utils, decoding, localize, selectivity, spikes,
               spike_distance, stats, viz)  # NOQA

from .viz import plot_shaded, plot_raster, plot_spikes  # NOQA
from .spikes import SpikeEpochs, Spikes  # NOQA
from .selectivity import depth_of_selectivity, explained_variance  # NOQA

__version__ = '0.3.dev0'
