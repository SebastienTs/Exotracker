from napari import Viewer
from napari.types import LayerDataTuple, PointsData, TracksData
from napari.layers import Image
from magicgui import magicgui
from tifffile import imread, TiffFile
from skimage.filters import gaussian
from scipy import ndimage
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import groupby
import ctypes
import math
from settings import filename, filename2, frame_timestep, proteins, skipfirst, skiplast, mx_prefrc, mx_postfrc

#### Helper functions (intensity measurement)

# Return coordinates inside a disk centered at (0,0)
def disk_pts(rad):
  pts = []
  rad = np.round(rad).astype(int)
  for x in range(-rad,rad+1,1):
    for y in range(-rad,rad+1,1):
      if (x**2+y**2) <= rad**2:
        pts.append([0, y, x])
  return pts


# Compute min/mean/max intensity within disks (at coordinates pts)
def disk_int_stats(img, pts, rad):
  rad = np.round(rad).astype(int)
  pts = np.round(pts).astype(int)
  offsets = disk_pts(rad)
  min_int,  mean_int, max_int = (np.full(len(pts), np.NaN), np.full(len(pts), np.NaN), np.full(len(pts), np.NaN))
  for i, pt in enumerate(pts):
    if (pt[1] >= rad) and (pt[2] >= rad) and (pt[1] < img.shape[1]-rad) and (pt[2] < img.shape[2]-rad):
        coords = np.round(np.array(pt)+np.array(offsets)).astype(int)
        min_int[i] = np.min(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        mean_int[i] = np.mean(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        max_int[i] = np.max(img[coords[:, 0], coords[:, 1], coords[:, 2]])
  return {'min': min_int, 'mean': mean_int, 'max': max_int, 'area': len(offsets)}

def hysteresis_burst_detector(int_profile, thresh_hi, thresh_lo):
    bursts = []
    in_burst = False
    start = 0
    for i, x in enumerate(int_profile):
        if not in_burst and x >= thresh_hi:
            in_burst = True
            start = i
        elif in_burst and x < thresh_lo:
            in_burst = False
            bursts.append((start, i, np.mean(int_profile[start:i])))
    # Complete last burst if needed
    if in_burst:
        bursts.append((start, len(int_profile)-1, np.mean(int_profile[start:len(int_profile)-1])))
    return bursts

# Median filter an intensity profile and return the longest/latest plateau above threshold
def estimate_track_lgth(int_profile, medrad, thr, mode):
    vals = np.array(int_profile)
    vals_flt = medfilt(vals, kernel_size=2*medrad+1)
    vals_flt = (vals_flt-vals_flt.min())/(vals_flt.max()-vals_flt.min())
    bursts = hysteresis_burst_detector(vals_flt, thr, thr)
    #if mode[0] == 'last':
    #    start, end, mean = bursts[-1]
    if mode[0] == 'longest':
        longest = 0
        stcroped = 0
        for i, burst in enumerate(bursts):
            st, en, _ = burst
            if en-st > longest or stcroped:
                start, end, mean = burst
                longest = en-st
                stcroped = (start==0)
    if mode[0] == 'highest':
        highest = 0
        stcroped = 0
        for i, burst in enumerate(bursts):
            st, en, mn = burst
            if mn > highest or stcroped:
                start, end, _ = burst
                highest = mn
                stcroped = (start == 0)
    return int(start), int(end), int(end-start)

#### Helper functions (track dataframes)

# Fill gaps in a track dataframe
def interpolate_track(group):
    frames = pd.RangeIndex(group['frame'].min(), group['frame'].max() + 1)
    group_interp = group.set_index('frame', drop=False).reindex(frames)
    group_interp['particle'] = group_interp['particle'].ffill()
    group_interp[['x', 'y', 'frame']] = group_interp[['x', 'y', 'frame']].interpolate(method='linear')
    return group_interp


# Add extra frames before a track dataframe (same position as track first frame)
def extend_dataframe_frames_pre(df, nrows):
    for j in range(nrows):
        df = pd.concat([pd.DataFrame(df.iloc[0]).T, df], ignore_index=True)
        df.loc[df.index[0], 'frame'] -= 1
    return df


# Append extra frames after a track dataframe (same position as track last frame)
def extend_dataframe_frames_post(df, nrows):
    for j in range(nrows):
        df = pd.concat([df, pd.DataFrame(df.iloc[-1]).T], ignore_index=True)
        df.loc[df.index[-1], 'frame'] += 1
    return df


# Flag tracks which distance to another track is below min_dst
def flag_min_dist(df, min_dst):
    key_to_ind = {group_key:i for i, (group_key, group_df) in enumerate(df.groupby('particle'))}
    keep = np.ones(len(key_to_ind), dtype=bool)
    for _, group in df.groupby('frame'):
        coords = group[['x', 'y']].values
        group_indices = group['particle'].values
        dist_matrix = np.sqrt(((coords[:, np.newaxis] - coords) ** 2).sum(axis=2))
        close_pairs = np.where((dist_matrix < min_dst) & (dist_matrix > 0))
        for elem in close_pairs[0]:
            keep[key_to_ind[group_indices[elem]]] = False
    return keep


#### Helper functions (napari layers)

def viewer_reset(vw: Viewer):

    for name in [layer.name for layer in vw.layers if not isinstance(layer, Image)]:
        vw.layers.remove(name)
    vw.layers.clear()

# Check if a layer with a specific name exists
def viewer_is_layer(vw: Viewer, layername):
    found = False
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername: found = True
    return found


# Close a layer with specific name
def viewer_close_layer(vw: Viewer, layername):
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername:
                vw.layers.pop(i)


#### Helper functions (miscellaneous)

# Add a subkey to a dictionnary entry
def acc_dict(dct, key, subkey, value):
    if key not in dct:
        dct[key] = {}
    dct[key][subkey] = value
    return dct

# Tile all opened matplotlib plots
def tile_windows(wdth, hght):
    figs = plt.get_fignums()
    n = len(figs)
    cols = math.ceil(math.sqrt(n))
    for i, num in enumerate(figs):
        plt.figure(num)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(i % cols * wdth,50 + i // cols * round(hght * 1.1), wdth, hght)


#### Napari dialog box

# Display message dialog box
def dialogboxmes(message, title):
    return ctypes.windll.user32.MessageBoxW(0, title, message, 0)


#### Track measurements analysis and plots (graphs.py script only)

# Normalized 0-centered logistic function
def logistic(x, steepness=4):
    return 1 / (1 + np.exp(-steepness*x))

## Multi-logistic function
def model_logistic(x, x1, h1, x2, h2, x3, h3):
    result = h1*logistic(x-x1)+h2*logistic(x-x2)+h3*logistic(x-x3)
    return result
