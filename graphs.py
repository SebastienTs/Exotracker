import math
import pickle
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from magicgui import magicgui
from pathlib import Path
from collections import defaultdict
from settings import proteins
from algos import *
warnings.filterwarnings("ignore")

## Fit a 2 up-step, 1 down-step logistic function to C2 intensity profiles
def fit_plateaus(tracks_props, proteins, first_trck, last_trck):

    # Close all plots
    plt.close('all')

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            if tracks_props[key]['protein1'] == proteins[0] and tracks_props[key]['protein2'] == proteins[1] and first_trck <= cnt <= last_trck:
                plt.figure()
                int_c2 = tracks_props[key]['ch2_ext_int']
                int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2))
                x = np.arange(0, len(int_c2))
                popt, _ = curve_fit(model_logistic, x, int_c2, maxfev=1000, bounds=([0, 0, 0, 0, len(int_c2)*0.8, -1], [len(int_c2), 1, len(int_c2), 1, len(int_c2), 0]))
                inds = np.argsort([popt[0], popt[2], popt[4]])
                xpos = np.array([popt[0], popt[2], popt[4]])
                xpos = np.round(xpos[inds]*100)/100
                plt.plot(x, int_c2, label='Data')
                plt.plot(x, model_logistic(x, *popt), 'r-', label='Fit')
                plt.vlines(x=xpos, ymin=0, ymax=1, colors='green', linestyles='dashed')
                plt.xlim(0, len(int_c2))
                plt.title(f'C2 Track {cnt} from {proteins[1]}')
            cnt += 1
    tile_windows(300, 200)
    plt.show()

# Plot filtered intensity profiles of C1-C2 track pairs
def  plot_tracks_intensity(tracks_props, tracks_c2_times, proteins, first_trck, last_trck, medrad, int_norm):

    # Close all plots
    plt.close('all')

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein1'] == proteins[0] and tracks_props[key]['protein2'] == proteins[1] and tracks_props[key]['ch2_positive'] == 1:
            if first_trck <= cnt <= last_trck:
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
                plt.title(f'C2 Track {cnt} from {proteins[1]}')
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)


# Plot C1/C2 averaged intensity time profiles + std
def plot_tracks_avg_intensity(tracks_props, proteins, medrad, int_norm):

    # C1 track intensity profile is resampled to a vector of fixed size (rsplgth)
    # C2 buffer is larger to accomodate a shift (rspgth) and pre-/post- frames (assumed not exceeding 3x C1 track length)
    rsplgth = 256
    arr_int_c1 = np.full((int(1*rsplgth), len(tracks_props)), np.nan)
    arr_int_c2 = np.full((int(4*rsplgth), len(tracks_props)), np.nan)
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein1'] == proteins[0] and tracks_props[key]['protein2'] == proteins[1] and tracks_props[key]['ch2_positive'] == 1:
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
    plt.plot(np.arange(0, 1, 1/rsplgth), avg_int_c1, color='red', label=proteins[0])
    plt.plot(np.arange(-1, 3, 1/rsplgth), avg_int_c2, color='green', label=proteins[1])
    fig.gca().fill_between(np.arange(0, 1, 1/rsplgth), avg_int_c1-std_int_c1, avg_int_c1+std_int_c1, color='red', alpha=0.2)
    fig.gca().fill_between(np.arange(-1, 3, 1/rsplgth), avg_int_c2-std_int_c2, avg_int_c2+std_int_c2, color='green', alpha=0.2)
    plt.xlim(-mx_prefrc, 1+mx_postfrc)
    plt.legend()
    plt.title('Average intensity profiles (N='+str(cnt-1)+')')
    plt.show(block=False)


# Plot proteins timelines (mean start/stop + std)
def plot_tracks_timelines(data_list, proteins):
    cols = ['red', 'green', 'blue', 'orange', 'pink', 'cyan', 'yellow']
    n = len(data_list)
    figsize = (6,3*n)
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    for i, data in enumerate(data_list):
        # Compute average +std track start / end
        start = np.mean(data[0])
        start_std = np.std(data[0])
        stop = np.mean(data[1])
        stop_std = np.std(data[1])
        if i==0:
            startref = start
            stopref = stop
        if i>0:
            print(f'{proteins[i]} mean track start shift: {start-startref:.2f} +/- {start_std:.2f}')
            print(f'{proteins[i]} mean track stop shift: {stop-stopref:.2f} +/- {stop_std:.2f}')
        # Plot timelines
        rect = Rectangle((start, n-1-i), stop-start, 1, facecolor=cols[i%8], alpha=0.25, label=proteins[i]+f'(N = {len(data[0])})')
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


@magicgui(call_button='Analyze Saved Tracks',
          groupfiles={'widget_type': 'Checkbox', 'tooltip': 'Process all .pkl files from current image folder'},
          model={'widget_type': 'Checkbox', 'tooltip': 'Model C2 tracks intensity profiles', 'visible': False},
          plot_intprofiles={'widget_type': 'Checkbox', 'tooltip': 'Plot intensity profiles'},
          plot_first_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'First track to plot'},
          plot_last_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'Last track to plot'},
          plot_avg_intprofile={'widget_type': 'Checkbox', 'tooltip': 'Plot average intensity profile'},
          plot_timelines={'widget_type': 'Checkbox', 'tooltip': 'Plot timelines'})
def graph_tracks(groupfiles=False, model=False, plot_intprofiles=False, plot_first_trck=1, plot_last_trck=25,
                 plot_avg_intprofile=True, plot_timelines=True):

    # Proteins of the current dataset
    proteins_str = load_images_tiff.proteins.value
    stripped = proteins_str[1:-1].replace("'", "")
    proteins = [x.strip() for x in stripped.split(',')]

    # Results files: same names as input images but .pkl extension
    files_C1 = [str(load_images_tiff.imagepath.value).replace('.tif', '.pkl')]
    files_C2 = [str(load_images_tiff.imagepath2.value).replace('.tif', '.pkl')]

    # Retrieve all results files in current C1/C2 image folder(s)
    if groupfiles:
        files_C1 = [str(f) for f in Path(files_C1[0]).parent.glob(f'*{".pkl"}') if f.is_file() and '_C1' in f.name]
        files_C2 = [str(f) for f in Path(files_C2[0]).parent.glob(f'*{".pkl"}') if f.is_file() and '_C2' in f.name]

    # Display file information
    print('---------------- Data Analysis ----------------')
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
    print(f'Total number of C2 positive C1 tracks: {len(tracks_c2_times.items())}')

    if model:
        fit_plateaus(tracks_props, proteins=proteins, first_trck=plot_first_trck, last_trck=plot_last_trck)
    if plot_intprofiles:
        plot_tracks_intensity(tracks_props, tracks_c2_times, proteins=proteins, first_trck=plot_first_trck, last_trck = plot_last_trck, medrad = 4, int_norm = False)
    if plot_avg_intprofile:
        medrad = analyze_tracks_int_gate.track_c2_int_medrad.value
        plot_tracks_avg_intensity(tracks_props, proteins=proteins, medrad=medrad, int_norm = True)

    if plot_timelines:

        # Scan for all proteins present in pkl files of current image
        proteins1 = set()
        proteins2 = set()
        for key, trck_times in tracks_c2_times.items():
            if tracks_props[key]['ch2_positive'] == 1:
                proteins1.add(tracks_props[key]['protein1'])
                proteins2.add(tracks_props[key]['protein2'])
        print(f'Analyzing timelines from proteins:')
        print(proteins1, proteins2)

        # Gather measurements
        starts_c1, starts_c2, ends_c1, ends_c2, lgths_c1, lgths_c2 = [defaultdict(list) for _ in range(6)]
        for key, trck_times in tracks_c2_times.items():
            if tracks_props[key]['ch2_positive'] == 1:
                ts = tracks_props[key]['frame_timestep']
                lgths_c1[tracks_props[key]['protein1']].append(tracks_props[key]['length']*ts)
                starts_c2[tracks_props[key]['protein2']].append(
                    (trck_times[0] - tracks_props[key]['int_preframe'])*ts)
                ends_c2[tracks_props[key]['protein2']].append((trck_times[1]-tracks_props[key]['int_preframe'])*ts)
                lgths_c2[tracks_props[key]['protein2']].append(trck_times[2]*ts)
        data_list, proteins = [], []
        for protein1 in proteins1:
            if protein1 in lgths_c1:
                print(
                    f'{protein1} mean track length: {np.mean(lgths_c1[protein1]):.2f} +/- {np.std(lgths_c1[protein1]):.2f} [range: {np.min(lgths_c1[protein1]):.2f} - {np.max(lgths_c1[protein1]):.2f}]')
                data_list.append([np.zeros_like(lgths_c1[protein1]), lgths_c1[protein1]])
                proteins.append(protein1)
        for protein2 in proteins2:
            if protein2 in lgths_c2:
                print(
                    f'{protein2} mean track length: {np.mean(lgths_c2[protein2]):.2f} +/- {np.std(lgths_c2[protein2]):.2f} [range: {np.min(lgths_c2[protein2]):.2f} - {np.max(lgths_c2[protein2]):.2f}]')
                data_list.append([starts_c2[protein2], ends_c2[protein2]])
                proteins.append(protein2)

        # Call plotting function
        plot_tracks_timelines(data_list, proteins=proteins)
