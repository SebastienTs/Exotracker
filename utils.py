from napari import Viewer
from napari.types import LayerDataTuple, PointsData, TracksData
from magicgui import magicgui
from tifffile import imread
from skimage.filters import gaussian
from scipy import ndimage
import pandas as pd
import numpy as np
import ctypes
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import groupby
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from settings import filename, filename2, start_frame, end_frame, c2_preframes, c2_postframes

#### Helper functions (intensity measurement)

# Return z,y,x coordinates of the inner points of a disk (centered at origin)
def disk_pts(rad):
  pts, ptsq = [], [[], [], [], []]
  rad = np.round(rad).astype(int)
  for x in range(-rad,rad+1,1):
    for y in range(-rad,rad+1,1):
      if (x**2+y**2) <= rad**2:
        pts.append([0, y, x])
        ptsq[2*(x>0)+(y>0)].append([0, y, x])
  return pts, ptsq


# Compute min/mean/max image intensity within disks of radius rad centered at coordinates pts
def disk_int_stats(img, pts, rad):
  rad = np.round(rad).astype(int)
  pts = np.round(pts).astype(int)
  offsets, offsetsq = disk_pts(rad)
  min_int,  mean_int, max_int = (np.full(len(pts), np.NaN), np.full(len(pts), np.NaN),
                                 np.full(len(pts), np.NaN))
  for i, pt in enumerate(pts):
    if (pt[1] >= rad) and (pt[2] >= rad) and (pt[1] < img.shape[1]-rad) and (pt[2] < img.shape[2]-rad):
        coords = np.round(np.array(pt)+np.array(offsets)).astype(int)
        min_int[i] = np.min(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        mean_int[i] = np.mean(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        max_int[i] = np.max(img[coords[:, 0], coords[:, 1], coords[:, 2]])
  return {'min': min_int, 'mean': mean_int, 'max': max_int, 'area': len(offsets)}


# Find the longest high level plateau (above thr) in a filtered intensity profile
def estimate_track_lgth(int_profile, medrad, thr):
    vals = np.array(int_profile)
    vals_flt = medfilt(vals, kernel_size=medrad)
    vals_flt = (vals_flt - vals_flt.min()) / (vals_flt.max() - vals_flt.min())
    trck_flag = (vals_flt >= thr).astype(int)
    #trck_flag = medfilt(trck_flag, kernel_size=2*medrad+1)
    start, end , lgth = longest_non_zero_sequence(trck_flag)
    return int(start), int(end), int(lgth), vals_flt, trck_flag

#### Helper functions (track dataframes)

# Fill gaps in a dataframe track
def interpolate_track(group):
    frames = pd.RangeIndex(group['frame'].min(), group['frame'].max() + 1)
    group_interp = group.set_index('frame', drop=False).reindex(frames)
    group_interp['particle'] = group_interp['particle'].ffill()
    group_interp[['x', 'y', 'frame']] = group_interp[['x', 'y', 'frame']].interpolate(method='linear')
    return group_interp


# Append extra frames after a dataframe track (same position as last frame)
def extend_dataframe_frames_post(df, nrows):
    for j in range(nrows):
        df = pd.concat([df, pd.DataFrame(df.iloc[-1]).T], ignore_index=True)
        df.loc[df.index[-1], 'frame'] += 1
    return df

# Append extra frames before a dataframe track (same position as first frame)
def extend_dataframe_frames_pre(df, nrows):
    for j in range(nrows):
        df = pd.concat([pd.DataFrame(df.iloc[0]).T, df], ignore_index=True)
        df.loc[df.index[0], 'frame'] -= 1
    return df

# Flag tracks with distance to another track < min_dst in a track dataframe
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

# Check if viewer layer with specific name exists
def viewer_is_layer(vw: Viewer, layername):
    found = False
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername: found = True
    return found


# Close all viewer layers holding the string layername in their name
def viewer_close_layer(vw: Viewer, layername):
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername:
                vw.layers.pop(i)

#### Helper functions (miscellaneous)

# Add a subkey to an entry of a dictionary
def acc_dict(dct, key, subkey, value):
    if key not in dct:
        dct[key] = {}
    dct[key][subkey] = value
    return dct

# Find the longest pulse in a binary sequence
def longest_non_zero_sequence(v):
    groups = [(k, len(list(g))) for k, g in groupby(v)]
    if not groups or not any(k for k, _ in groups):
        return 0, 0, 0
    i, l = max(((i, l) for i, (k, l) in enumerate(groups) if k), key=lambda x: x[1])
    st, ed = sum(g[1] for g in groups[:i]), sum(g[1] for g in groups[:i]) + l - 1
    return st, ed, ed-st

# Tile multiple matplotlib plots
def tile_windows(wdth, hght):
    figs = plt.get_fignums()
    n = len(figs)
    cols = math.ceil(math.sqrt(n))
    for i, num in enumerate(figs):
        plt.figure(num)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(i % cols * wdth,50 + i // cols * round(hght * 1.1), wdth, hght)


#### Napari widgets and dialog box

## TIFF images loader widget
@magicgui(call_button='Load',
          imagepath={'widget_type': 'FileEdit', 'label': 'Chan1'},
          imagepath2={'widget_type': 'FileEdit', 'label': 'Chan2'},
          start_frame={'widget_type': 'IntSlider', 'max': 999, 'tooltip': 'Skip first frames to avoid instabilities'},
          end_frame={'widget_type': 'IntSlider', 'max': 999, 'tooltip': 'Skip last frames to avoid bleaching'})
def load_images_tiff(vw:Viewer, imagepath=filename, imagepath2=filename2, start_frame=start_frame, end_frame=end_frame):
    img = imread(imagepath, key=range(start_frame, end_frame)).astype(np.uint16)

    if str(imagepath2).endswith('.tif'):
        img2 = imread(imagepath2, key=range(start_frame, end_frame)).astype(np.uint16)
        #img2_corr = ((img2 + img2[0, :, :].max() + 1) - gaussian(img2[0, :, :], sigma=5, preserve_range=True)).astype(np.uint16)
        if viewer_is_layer(vw, 'Channel2'):
            vw.layers['Channel2'].data = img2
            #vw.layers['Channel2_corr'].data = img2_corr
        else:
            vw.add_image(img2, name='Channel2')
            #vw.add_image(img2_corr, name='Channel2_corr')
        print(f'Loaded image {imagepath2} ({img2.shape})')

    if viewer_is_layer(vw, 'Channel1'):
        vw.layers['Channel1'].data = img
    else:
        vw.add_image(img, name='Channel1')
    print(f'Loaded image {imagepath} ({img.shape})')

    return None


# Display message dialog box
def dialogboxmes(message, title):
    return ctypes.windll.user32.MessageBoxW(0, title, message, 0)


#### Track measurements analysis and plots (graphs.py script only)

# Re-analyze and plot C2 tracks (possibly multiple per figure)
def analyze_and_plot_C2_tracks_intensity(tracks_props, trckperplot, medrad, trck_thr):
    cnt = 1
    plt.figure()
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            start, end, _ , vals_flt, trck_flag = estimate_track_lgth(tracks_props[key]['ch2_ext_int'], medrad, trck_thr)
            plt.plot(vals_flt, linestyle=':')
            vals_flt[1:start] = np.NaN
            vals_flt[end:] = np.NaN
            plt.plot(vals_flt, color=plt.gca().get_lines()[-1].get_color())
            if cnt%trckperplot == 0:
                plt.figure()
            cnt += 1
    tile_windows(300, 200)
    plt.show()

# Normalized 0-centered logistic function
def logistic(x, steepness=4):
    return 1 / (1 + np.exp(-steepness*x))

## Multi-logistic function
def model_logistic(x, x1, h1, x2, h2, x3, h3):
    result = h1*logistic(x-x1)+h2*logistic(x-x2)+h3*logistic(x-x3)
    return result

## Fit a 2 up-step, 1 down-step logistic function to C2 normalized extended intensity profiles
def model_C2_tracks_intensity(tracks_props, timestep):
    plt.figure()
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            int_c2 = tracks_props[key]['ch2_ext_int']
            int_c2 = (int_c2 - min(int_c2)) / (max(int_c2) - min(int_c2))
            x = np.arange(0, len(int_c2)*timestep, timestep)
            popt, _ = curve_fit(model_logistic, x, int_c2, maxfev=10000,
                             bounds=([0, 0, 0, 0, len(int_c2)*timestep*0.8, -1],
                                     [len(int_c2)*timestep, 1, len(int_c2)*timestep, 1, len(int_c2)*timestep, 0]))
            inds = np.argsort([popt[0], popt[2], popt[4]])
            xpos = np.array([popt[0], popt[2], popt[4]])
            xpos = np.round(xpos[inds]*100)/100
            plt.plot(x, int_c2, label='Data')
            plt.plot(x, model_logistic(x, *popt), 'r-', label='Fit')
            plt.vlines(x=xpos, ymin=0, ymax=1, colors='green', linestyles='dashed')
            plt.xlim(0, len(int_c2)*timestep)
            plt.show(block=False)

# Plot C1-C2 track pairs intensity profiles
def plot_C1_C2_tracks_intensity(tracks_props, tracks_c2_times, mx_trck, medrad, int_norm):
    cnt = 1
    plt.figure()
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1 and cnt <= mx_trck:
            # Median filter
            #int_c1 = medfilt(tracks_props[key]['ch1_int'], kernel_size=2*medrad+1)
            int_c1 = medfilt(tracks_props[key]['ch1_ext_int'], kernel_size=2*medrad+1)
            int_c2 = medfilt(tracks_props[key]['ch2_ext_int'], kernel_size=2*medrad+1)
            # Normalize intensity
            if int_norm:
                int_c1 = (int_c1 - min(int_c1)) / (max(int_c1) - min(int_c1))
                int_c2 = (int_c2 - min(int_c2)) / (max(int_c2) - min(int_c2))
            # Plots
            plt.plot(int_c1, linestyle='--', color='red')
            int_c1[:c2_preframes], int_c1[-c2_preframes:] = np.NaN, np.NaN
            plt.plot(int_c1, color='red')
            plt.plot(int_c2, linestyle=':', color='green')
            start, end, lgth = tracks_c2_times[key][0], tracks_c2_times[key][1], tracks_c2_times[key][2]
            int_c2[1:start], int_c2[end:] = np.NaN, np.NaN
            plt.plot(int_c2, color='green')
            #plt.plot(int_c2, color=plt.gca().get_lines()[-1].get_color())
            plt.figure()
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)


# Plot C1/C2 averaged intensity time profiles + std
def plot_C1_C2_tracks_avg_intensity(tracks_props, mx_trck, mx_prefrc, mx_postfrc, medrad, int_norm, proteins):
    rsplgth = 256
    arr_int_c1 = np.full((int(1*rsplgth), len(tracks_props)), np.nan)
    arr_int_c2 = np.full((int(4*rsplgth), len(tracks_props)), np.nan)
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1 and cnt <= mx_trck:
            # Retrieve intensity profiles
            int_c1 = tracks_props[key]['ch1_int']
            int_c2 = tracks_props[key]['ch2_ext_int']
            # Median filter intensity profiles
            int_c1 = medfilt(int_c1, kernel_size=2*medrad+1)
            int_c2 = medfilt(int_c2, kernel_size=2*medrad+1)
            # Resample intensity profiles (normalize C1 track length to rsmlgth)
            c1_lgth = len(int_c1)
            c2_lgth = len(int_c2)
            int_c1 = np.interp(np.linspace(0, 1, num=rsplgth), np.linspace(0, 1, num=c1_lgth), int_c1)
            int_c2 = np.interp(np.linspace(0, 1, num=int(c2_lgth/c1_lgth*rsplgth)), np.linspace(0, 1, num=c2_lgth), int_c2)
            # Normalize intensity
            if int_norm:
                int_c1 = (int_c1-min(int_c1))/(max(int_c1)-min(int_c1))
                int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2))
            arr_int_c1[:len(int_c1), cnt-1] = int_c1
            # Account for possible preframe shifting of C2 (C2 has fixed shift of rsplgth and variable shift -preshift)
            preshift = int(c2_preframes/c1_lgth*rsplgth)
            arr_int_c2[rsplgth-preshift:rsplgth-preshift+len(int_c2), cnt-1] = int_c2
            # Set C2 pre/post intensity to first/last intensity value
            arr_int_c2[:rsplgth-preshift,cnt-1] = arr_int_c2[rsplgth-preshift, cnt-1]
            arr_int_c2[rsplgth-preshift+len(int_c2):, cnt-1] = arr_int_c2[rsplgth-preshift+len(int_c2)-1, cnt-1]
            cnt += 1

    # Compute intensity profiles statistics
    avg_int_c1, avg_int_c2 = np.nanmean(arr_int_c1, axis=1), np.nanmean(arr_int_c2, axis=1)
    std_int_c1, std_int_c2 = np.nanstd(arr_int_c1, axis=1), np.nanstd(arr_int_c2, axis=1)

    # Plot graphs
    fig = plt.figure()
    plt.title("Intensity profiles (time and intensity normalized)")
    plt.plot(np.arange(0, 1, 1/rsplgth), avg_int_c1, color='red', label=proteins[0])
    plt.plot(np.arange(-1, 3, 1/rsplgth), avg_int_c2, color='green', label=proteins[1])
    fig.gca().fill_between(np.arange(0, 1, 1/rsplgth), avg_int_c1-std_int_c1, avg_int_c1+std_int_c1, color='red', alpha=0.2)
    fig.gca().fill_between(np.arange(-1, 3, 1/rsplgth), avg_int_c2-std_int_c2, avg_int_c2+std_int_c2, color='green', alpha=0.2)
    plt.xlim(-mx_prefrc, 1+mx_postfrc)
    plt.legend()
    plt.show(block=False)

# Plot proteins timelines (mean start/stop + std)
def plot_timelines(data_list, proteins, mx_frame, timestep):
    cols = ['red', 'green', 'blue', 'orange', 'pink', 'cyan', 'yellow']
    n = len(data_list)
    figsize = (6,3*n)
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    for i, data in enumerate(data_list):
        # Compute average +std track start / end
        start = np.mean(data[0])*timestep
        start_std = np.std(data[0])*timestep
        stop = np.mean(data[1])*timestep
        stop_std = np.std(data[1])*timestep
        if i==0:
            startref = start
            stopref = stop
        if i>0:
            print(f'C2 mean track start shift: {start-startref:.2f} +/- {start_std:.2f}')
            print(f'C2 mean track stop shift: {stop - stopref:.2f} +/- {stop_std:.2f}')
        # Plot timelines
        rect = Rectangle( (start , n-1-i), stop-start, 1, facecolor=cols[i%8], alpha=0.25, label=proteins[i])
        ax.add_patch(rect)
        rect = Rectangle( (start-start_std, n-i-0.55), width=2*start_std, height=0.01, facecolor='black')
        ax.add_patch(rect)
        rect = Rectangle( (stop-stop_std, n-i-0.45), width=2*stop_std, height=0.01, facecolor='black')
        ax.add_patch(rect)
        ax.plot()
    plt.xlim(0, mx_frame * timestep)
    plt.xlabel('Time (s)')
    plt.ylim(0, n)
    ax.yaxis.set_visible(False)
    plt.legend()
    plt.show(block=False)
