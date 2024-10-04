import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# colour maps used for visualising data

# salinity
sea = to_rgb('midnightblue')
pond = to_rgb('darkblue')

cdict_phi = {'red':   ((0.0, 0.0, sea[0]),
                       (0.1, 0.2*pond[0], 0.2*pond[0]),
                       (1.0, pond[0], 0.0)),
             
             'green': ((0.0, 0.0, sea[1]),
                       (0.1, 0.2*pond[1], 0.2*pond[1]),
                       (1.0, pond[1], 0.0)),
             
             'blue':  ((0.0, 0.0, sea[2]),
                       (0.1, 0.2*pond[2], 0.2*pond[2]),
                       (1.0, pond[2], 0.0))
             }

salinity = LinearSegmentedColormap('salinity', cdict_phi)
mpl.colormaps.register(cmap = salinity)

#mpl.colormaps.unregister('salinity')


# solid fraction
sea = to_rgb('midnightblue')
ice = to_rgb('lightsteelblue')
#ice = to_rgb('grey')

cdict_phi = {'red':   ((0.0, 0.0, sea[0]),
                       (0.1, 0.2*ice[0], 0.2*ice[0]),
                       (1.0, ice[0], 0.0)),
             
             'green': ((0.0, 0.0, sea[1]),
                       (0.1, 0.2*ice[1], 0.2*ice[1]),
                       (1.0, ice[1], 0.0)),
             
             'blue':  ((0.0, 0.0, sea[2]),
                       (0.1, 0.2*ice[2], 0.2*ice[2]),
                       (1.0, ice[2], 0.0))
             }

cdict_phi['alpha'] = ((0.0, 0.0, 0.0),
                    (0.0001, 0.0, 0.9), 
#                    (0.05, 0.0, 0.9), 
#                    (0.4, 0.1, 0.5), 
                    (1.0, 1.0, 1.0))

sea_ice = LinearSegmentedColormap('SeaIce', cdict_phi)
mpl.colormaps.register(cmap = sea_ice)

#mpl.colormaps.unregister('SeaIce')



# salinity (for plots without solid fraction)
pond = to_rgb('xkcd:sea blue')
channel = to_rgb('greenyellow')
base = to_rgb('xkcd:banana yellow')
sea = to_rgb('goldenrod')


cdict_C = {'red':   ((0.0, 0.0, pond[0]),
                       (0.4, channel[0], channel[0]),
                       (0.6, base[0], base[0]),
#                       (0.1, 0.2*pond[0], 0.2*pond[0]),
                       (1.0, sea[0], 0.0)),
             
             'green': ((0.0, 0.0, pond[1]),
                       (0.4, channel[1], channel[1]),
                       (0.6, base[1], base[1]),
#                       (0.1, 0.2*pond[1], 0.2*pond[1]),
                       (1.0, sea[1], 0.0)),
             
             'blue':  ((0.0, 0.0, pond[2]),
                       (0.4, channel[2], channel[2]),
                       (0.6, base[2], base[2]),
#                       (0.1, 0.2*pond[2], 0.2*pond[2]),
                       (1.0, sea[2], 0.0))
             }

salinity_alt = LinearSegmentedColormap('Sal', cdict_C)
mpl.colormaps.register(cmap = salinity_alt)

#mpl.colormaps.unregister('Sal')
