import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection

def plotcolored(ax,x,y,c,*,lw=None,cmap=None,norm=None):
    if lw is None:
        lw = mpl.rcParams['lines.linewidth']
    if cmap is None:
        cmap = mpl.rcParams['image.cmap']
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0,vmax=1)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc
