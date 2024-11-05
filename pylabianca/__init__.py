from . import (io, utils, decoding, localize, selectivity, spikes,
               analysis, spike_distance, stats, viz)  # NOQA

from .analysis import (
    aggregate, dict_to_xarray, xarray_to_dict
)  # NOQA
from .viz import plot_shaded, plot_raster, plot_spikes  # NOQA
from .spikes import SpikeEpochs, Spikes  # NOQA
from .selectivity import depth_of_selectivity, explained_variance  # NOQA

__version__ = '0.4.dev0'
