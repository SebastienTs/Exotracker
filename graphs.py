import math
import pickle
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from magicgui import magicgui
from pathlib import Path
from settings import frame_timestep, proteins
from algos import *
warnings.filterwarnings("ignore")

# Re-analyze (filtering + longuest plateau search) and plot C2 tracks
def reanalyze_tracks(tracks_props, protein, medrad, trck_thr, first_trck, last_trck):

    # Close all plots
    plt.close('all')

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein2'] == protein and tracks_props[key]['ch2_positive'] == 1:
            if cnt >= first_trck and cnt <= last_trck:
                plt.figure()
                start, end, _ , vals_flt, _ = estimate_track_lgth(tracks_props[key]['ch2_ext_int'], medrad, trck_thr)
                plt.plot(vals_flt, linestyle=':')
                vals_flt[1:start] = np.NaN
                vals_flt[end:] = np.NaN
                plt.plot(vals_flt, color=plt.gca().get_lines()[-1].get_color())
                plt.grid()
                plt.title(f'C2 Track {cnt} from {protein}')
            cnt += 1
    tile_windows(300, 200)
    plt.show()

## Fit a 2 up-step, 1 down-step logistic function to C2 intensity profiles
def fit_plateaus(tracks_props, protein, timestep, first_trck, last_trck):

    # Close all plots
    plt.close('all')

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            if tracks_props[key]['protein2'] == protein and cnt >= first_trck and cnt <= last_trck:
                plt.figure()
                int_c2 = tracks_props[key]['ch2_ext_int']
                int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2))
                x = np.arange(0, len(int_c2)*timestep, timestep)
                popt, _ = curve_fit(model_logistic, x, int_c2, maxfev=1000, bounds=([0, 0, 0, 0, len(int_c2)*timestep*0.8, -1], [len(int_c2)*timestep, 1, len(int_c2)*timestep, 1, len(int_c2)*timestep, 0]))
                inds = np.argsort([popt[0], popt[2], popt[4]])
                xpos = np.array([popt[0], popt[2], popt[4]])
                xpos = np.round(xpos[inds]*100)/100
                plt.plot(x, int_c2, label='Data')
                plt.plot(x, model_logistic(x, *popt), 'r-', label='Fit')
                plt.vlines(x=xpos, ymin=0, ymax=1, colors='green', linestyles='dashed')
                plt.xlim(0, len(int_c2)*timestep)
                plt.title(f'C2 Track {cnt} from {protein}')
            cnt += 1
    tile_windows(300, 200)
    plt.show()

# Plot filtered intensity profiles of C1-C2 track pairs
def  plot_tracks_intensity(tracks_props, tracks_c2_times, protein, first_trck, last_trck, medrad, int_norm):

    # Close all plots
    plt.close('all')

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein2'] == protein and tracks_props[key]['ch2_positive'] == 1:
            if cnt >= first_trck and cnt <= last_trck:
                int_preframe = tracks_props[key]['int_preframe']
                int_postframe = tracks_props[key]['int_postframe']
                # Median filter
                int_c1 = medfilt(tracks_props[key]['ch1_ext_int'], kernel_size=2*medrad+1)
                int_c2 = medfilt(tracks_props[key]['ch2_ext_int'], kernel_size=2*medrad+1)
                #ctr_c2 = medfilt(tracks_props[key]['ch2_ext_contrast'], kernel_size=2*medrad+1)*1000
                # Normalize intensity
                if int_norm:
                    int_c1 = (int_c1-min(int_c1))/(max(int_c1)-min(int_c1))
                    int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2))
                # Plot
                plt.figure(cnt)
                plt.plot(int_c1, linestyle='--', color='red')
                int_c1[:int_preframe], int_c1[-int_postframe:] = np.NaN, np.NaN
                plt.plot(int_c1, color='red')
                plt.plot(int_c2, linestyle=':', color='green')
                #plt.plot(ctr_c2, linestyle=':', color='blue')
                start, end, lgth = tracks_c2_times[key][0], tracks_c2_times[key][1], tracks_c2_times[key][2]
                int_c2[1:start], int_c2[end:] = np.NaN, np.NaN
                plt.plot(int_c2, color='green')
                #plt.plot(ctr_c2, color='blue')
                plt.grid()
                plt.title(f'C2 Track {cnt} from {protein}')
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)


# Plot C1/C2 averaged intensity time profiles + std
def plot_tracks_avg_intensity(tracks_props, protein, medrad, int_norm):

    # C1 track intensity profile is resampled to a vector of fixed size (rsplgth)
    # C2 buffer is larger to accomodate a shift (rspgth) and pre-/post- frames (assumed not exceeding 3x C1 track length)
    rsplgth = 256
    arr_int_c1 = np.full((int(1*rsplgth), len(tracks_props)), np.nan)
    arr_int_c2 = np.full((int(4*rsplgth), len(tracks_props)), np.nan)
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein2'] == protein and tracks_props[key]['ch2_positive'] == 1:
            int_preframe = tracks_props[key]['int_preframe']
            # Load C1 and C2 intensity profiles
            int_c1 = tracks_props[key]['ch1_int']
            int_c2 = tracks_props[key]['ch2_ext_int']
            c1_lgth = len(int_c1)
            c2_lgth = len(int_c2)
            # Median filter intensity profiles
            int_c1 = medfilt(int_c1, kernel_size=2*medrad+1)
            int_c2 = medfilt(int_c2, kernel_size=2*medrad+1)
            # Normalize intensity profiles
            if int_norm:
                int_c1 = (int_c1-min(int_c1))/(max(int_c1)-min(int_c1))
                int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2))
            # Resample intensity profiles so that C1 intensity profile has fixed length (rsplgth samples)
            int_c1 = np.interp(np.linspace(0, 1, num=rsplgth), np.linspace(0, 1, num=c1_lgth), int_c1)
            int_c2 = np.interp(np.linspace(0, 1, num=round(c2_lgth/c1_lgth*rsplgth)), np.linspace(0, 1, num=c2_lgth), int_c2)
            # Store C1 intensity profile to a matrix holding all the aligned profiles
            arr_int_c1[:len(int_c1), cnt-1] = int_c1
            # Store C2 intensity profile to a matrix holding all the aligned profiles (account for int_preframe shift)
            preshift = int(int_preframe/c1_lgth*rsplgth)
            arr_int_c2[rsplgth-preshift:rsplgth-preshift+len(int_c2), cnt-1] = int_c2
            # Pad C2 intensity to first/last intensity value
            arr_int_c2[:rsplgth-preshift, cnt-1] = arr_int_c2[rsplgth-preshift,cnt-1]
            arr_int_c2[rsplgth-preshift+len(int_c2):, cnt-1] = arr_int_c2[rsplgth-preshift+len(int_c2)-1, cnt-1]
            cnt += 1

    # Crop C1 and C2 matrices to the effective number of intensity profiles that were stored
    arr_int_c1 = arr_int_c1[:, 0:cnt - 1]
    arr_int_c2 = arr_int_c2[:, 0:cnt - 1]

    # Compute intensity statistics (use nanmean/nanstd if intensity values weren't padded!)
    avg_int_c1, avg_int_c2 = np.mean(arr_int_c1, axis=1), np.mean(arr_int_c2, axis=1)
    std_int_c1, std_int_c2 = np.std(arr_int_c1, axis=1), np.std(arr_int_c2, axis=1)

    # Plot
    fig = plt.figure()
    plt.plot(np.arange(0, 1, 1/rsplgth), avg_int_c1, color='red', label='exo84')
    plt.plot(np.arange(-1, 3, 1/rsplgth), avg_int_c2, color='green', label=protein)
    fig.gca().fill_between(np.arange(0, 1, 1/rsplgth), avg_int_c1-std_int_c1, avg_int_c1+std_int_c1, color='red', alpha=0.2)
    fig.gca().fill_between(np.arange(-1, 3, 1/rsplgth), avg_int_c2-std_int_c2, avg_int_c2+std_int_c2, color='green', alpha=0.2)
    plt.xlim(-mx_prefrc, 1+mx_postfrc)
    plt.legend()
    plt.title('Average intensity profiles (N='+str(cnt-1)+')')
    plt.show(block=False)


# Plot proteins timelines (mean start/stop + std)
def plot_tracks_timelines(data_list, proteins, timestep):
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
            print(f'C2 mean track stop shift: {stop-stopref:.2f} +/- {stop_std:.2f}')
        # Plot timelines
        rect = Rectangle((start, n-1-i), stop-start, 1, facecolor=cols[i%8], alpha=0.25, label=proteins[i])
        ax.add_patch(rect)
        rect = Rectangle((start-start_std, n-i-0.55), width=2*start_std, height=0.01, facecolor='black')
        ax.add_patch(rect)
        rect = Rectangle((stop-stop_std, n-i-0.45), width=2*stop_std, height=0.01, facecolor='black')
        ax.add_patch(rect)
        ax.plot()
    plt.xlabel('Time (s)')
    plt.ylim(0, n)
    ax.yaxis.set_visible(False)
    plt.legend()
    plt.title('Track timelines')
    plt.show(block=False)


@magicgui(call_button='Analyze',
          groupfiles={'widget_type': 'Checkbox', 'tooltip': 'Process all .pkl files from current image folder'},
          protein={'widget_type': 'LineEdit', 'tooltip': 'C2 protein name (used as filter for intensity plots)'},
          model={'widget_type': 'Checkbox', 'tooltip': 'Model C2 tracks intensity profiles'},
          medrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9, 'tooltip': 'C2 track detection median filter window half length'},
          trck_thr={'widget_type': 'FloatSlider', 'min': 0.25, 'max': 0.5, 'step': 0.01, 'tooltip': 'C2 track detection normalized intensity threshold'},
          reanalyze={'widget_type': 'Checkbox', 'tooltip': 'Re-analyze C2 tracks based on intensity profiles'},
          plot_intprofiles={'widget_type': 'Checkbox', 'tooltip': 'Plot intensity profiles'},
          plot_first_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'First track to plot'},
          plot_last_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'Last track to plot'},
          plot_avg_intprofile={'widget_type': 'Checkbox', 'tooltip': 'Plot average intensity profile'},
          plot_timelines={'widget_type': 'Checkbox', 'tooltip': 'Plot timelines'})
def graph_tracks(groupfiles=False, protein='sec9', model=False, medrad=5, trck_thr=0.35, reanalyze=False,
                 plot_intprofiles=False, plot_first_trck=1, plot_last_trck=25, plot_avg_intprofile=True, plot_timelines=True):

    # Results files have same names as C1/C2 images but .pkl extension
    files_C1 = [str(load_images_tiff.imagepath.value).replace('.tif', '.pkl')]
    files_C2 = [str(load_images_tiff.imagepath2.value).replace('.tif', '.pkl')]

    # List results files in image folder
    if groupfiles:
        files_C1 = [str(f) for f in Path(files_C1[0]).parent.glob(f'*{".pkl"}') if f.is_file() and '_C1' in f.name]
        files_C2 = [str(f) for f in Path(files_C2[0]).parent.glob(f'*{".pkl"}') if f.is_file() and '_C2' in f.name]

    # Display analysis information
    print('---------------- Starting data analysis ----------------')
    print('Analyzing files: ')
    print([Path(file_C1).name for file_C1 in files_C1])
    print([Path(file_C2).name for file_C2 in files_C2])

    # Aggregate content from pkl files to dictionaries
    tracks_props = {}
    tracks_c2_times = {}
    for i, (file_C1, file_C2) in enumerate(zip(files_C1, files_C2)):
        with open(file_C1, 'rb') as file:
            tracks_props.update({str(k) + '_'+str(i): v for k, v in pickle.load(file).items()})
        with open(file_C2, 'rb') as file:
            tracks_c2_times.update({str(k) + '_'+str(i): v for k, v in pickle.load(file).items()})

    ## Statistics over all C1/C2 tracks
    print(f'Number of C2 positive C1 tracks: {len(tracks_c2_times.items())}')
    starts_c1, starts_c2 = [[] for _ in range(2)]
    ends_c1, ends_c2 = [[] for _ in range(2)]
    lgths_c1, lgths_c2 = [[] for _ in range(2)]
    for key, trck_times in tracks_c2_times.items():
        int_preframe = tracks_props[key]['int_preframe']
        lgths_c1.append(tracks_props[key]['length'])
        starts_c2.append(trck_times[0]-int_preframe)
        ends_c2.append(trck_times[1]-int_preframe)
        lgths_c2.append(trck_times[2])
    print(f'C1 mean track length: {np.mean(lgths_c1)*frame_timestep:.2f} +/- {np.std(lgths_c1)*frame_timestep:.2f} [range: {np.min(lgths_c1)*frame_timestep:.2f} - {np.max(lgths_c1)*frame_timestep:.2f}]')
    print(f'C2 mean track length: {np.mean(lgths_c2)*frame_timestep:.2f} +/- {np.std(lgths_c2)*frame_timestep:.2f} [range: {np.min(lgths_c2)*frame_timestep:.2f} - {np.max(lgths_c2)*frame_timestep:.2f}]')

    # Re-analyze C2 tracks
    if reanalyze:
        reanalyze_tracks(tracks_props, protein=protein, medrad=medrad, trck_thr=trck_thr, first_trck=plot_first_trck, last_trck=plot_last_trck)
    if model:
        fit_plateaus(tracks_props, protein=protein, timestep=frame_timestep, first_trck=plot_first_trck, last_trck=plot_last_trck)
    if plot_intprofiles:
        plot_tracks_intensity(tracks_props, tracks_c2_times, protein=protein, first_trck=plot_first_trck, last_trck = plot_last_trck, medrad = 4, int_norm = False)
    if plot_avg_intprofile:
        plot_tracks_avg_intensity(tracks_props, protein=protein, medrad=medrad, int_norm = True)
    if plot_timelines:
        data_list = [[np.zeros_like(lgths_c1), lgths_c1], [starts_c2, ends_c2]]
        plot_tracks_timelines(data_list, proteins=['exo84', 'sec9'], timestep=frame_timestep)
