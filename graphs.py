import re
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from magicgui import magicgui
from pathlib import Path
from collections import defaultdict
from os import path, makedirs
from algos import *
warnings.filterwarnings("ignore")

## Fit a 2 up-step, 1 down-step logistic function to C2 intensity profiles
def fit_plateaus(tracks_props, first_trck, last_trck):

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            if first_trck <= cnt <= last_trck:
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
                plt.title(f'C2 Track {cnt}')
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)

# Plot filtered intensity profiles of C1-C2 track pairs
def  plot_tracks_intensity(tracks_props, tracks_c2_times, first_trck, last_trck, int_norm):

    medrad = analyze_tracks_int_gate.track_c2_int_medrad.value

    # Plot loop
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            if first_trck <= cnt <= last_trck and tracks_props[key]['ignore_track'] == 0:
                track = tracks_props[key]['track']
                int_preframe = tracks_props[key]['int_preframe']
                int_postframe = tracks_props[key]['int_postframe']
                # Intensity profiles
                int_c1 = np.array(tracks_props[key]['ch1_ext_int'])
                int_c2 = np.array(tracks_props[key]['ch2_ext_int'])
                # Median filter
                int_c1 = medfilt(int_c1, kernel_size=2*medrad+1)
                int_c2 = medfilt(int_c2, kernel_size=2*medrad+1)
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
                xstart, ystart, fstart = track.iloc[0]['x'], track.iloc[0]['y'], track.iloc[0]['frame']
                plt.title(f'C2 Track {cnt} (T: {int(fstart-1)}, Y: {int(ystart)}, X: {int(xstart)})')
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)


# Plot C1/C2 averaged intensity time profiles + std
def plot_tracks_avg_intensity(tracks_props, proteins, int_norm):

    # C1 intensity profile is resampled to a fixed size vector arr_int_c1 (length rsplgth)
    # C2 buffer is four times larger than C1 buffer to accomodate:
    #   - pre-analysis (up to C1 duration by design since it cannot exceed min C1 duration)
    #   - post-analysis (up to C1 duration for plotting)
    #   - shift to align resampled intensity profiles due to pre-analysis (up to C1 duration by design)
    rsplgth = 256
    arr_int_c1 = np.full((int(1*rsplgth), len(tracks_props)), np.nan)
    arr_int_c2 = np.full((int(4*rsplgth), len(tracks_props)), np.nan)
    cnt, cnt2 = 0, 0
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein1'] == proteins[0] and tracks_props[key]['protein2'] == proteins[1] and tracks_props[key]['ch2_positive'] == 1:
            if tracks_props[key]['ignore_track'] == 0:
                int_preframe = tracks_props[key]['int_preframe']
                # Load C1 and C2 intensity profiles
                int_c1 = tracks_props[key]['ch1_int']
                int_c2 = tracks_props[key]['ch2_ext_int']
                c1_lgth = len(int_c1)
                c2_lgth = len(int_c2)
                # Median filter intensity profiles
                int_c1 = medfilt(int_c1, kernel_size=9)
                int_c2 = medfilt(int_c2, kernel_size=9)
                # Normalize intensity profiles
                if int_norm:
                    int_c1 = (int_c1-min(int_c1))/(max(int_c1)-min(int_c1))
                    int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2))
                # Resample intensity profiles so that C1 intensity profile has fixed length (rsplgth samples)
                int_c1 = np.interp(np.linspace(0, 1, num=rsplgth), np.linspace(0, 1, num=c1_lgth), int_c1)
                int_c2 = np.interp(np.linspace(0, 1, num=round(c2_lgth/c1_lgth*rsplgth)), np.linspace(0, 1, num=c2_lgth), int_c2)
                # Crop resampled C2 intensity profile if it exceeds 3x C1 duration (pre + post up to C1 duration)
                int_c2 = int_c2[0:min(len(int_c2), int(3*rsplgth))]
                # Store C1 intensity profile to a matrix holding all the aligned profiles
                arr_int_c1[:len(int_c1), cnt] = int_c1
                # Store C2 intensity profile to a matrix holding all the aligned profiles (account for int_preframe shift)
                preshift = int(int_preframe/c1_lgth*rsplgth)
                arr_int_c2[rsplgth-preshift:rsplgth-preshift+len(int_c2), cnt] = int_c2
                # Pad C2 intensity to first/last intensity value
                arr_int_c2[:rsplgth-preshift, cnt] = arr_int_c2[rsplgth-preshift,cnt]
                arr_int_c2[rsplgth-preshift+len(int_c2):, cnt] = arr_int_c2[rsplgth-preshift+len(int_c2)-1, cnt]
                cnt += 1
            else:
                cnt2 += 1
    print('---------------- Intensity Profiles ----------------')
    print(f'Number of C2 positive C1 tracks used for statistics: {cnt} ({cnt2} ignored)')

    # Crop C1 and C2 matrices to the effective number of intensity profiles that were stored
    arr_int_c1 = arr_int_c1[:, 0:cnt]
    arr_int_c2 = arr_int_c2[:, 0:cnt]

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
    plt.title('Average intensity profiles (N='+str(cnt)+')')
    plt.show(block=False)


# Plot proteins timelines (mean start/end + std)
def plot_tracks_timelines(data_list, proteins):
    cols = ['red', 'green', 'blue', 'orange', 'pink', 'cyan', 'yellow']
    n = len(data_list)
    figsize = (6,3*n)
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    for i, data in enumerate(data_list):
        # Compute average +std track start / end
        start = np.mean(data[0])
        start_std = np.std(data[0])
        end = np.mean(data[1])
        end_std = np.std(data[1])
        lgths = np.array(data[1])-np.array(data[0])
        if i==0:
            startref = start
            endref = end
            text = f'{np.mean(lgths):.1f} +/- {np.std(lgths):.1f}'
        if i>0:
            text = f'{start-startref:.2f} +/- {start_std:.1f}\n{np.mean(lgths):.1f} +/- {np.std(lgths):.1f}\n{end-endref:.2f} +/- {end_std:.1f}'
        # Plot timelines
        rectangle = Rectangle((start, n-1-i), end-start, 1, facecolor=cols[i%8], alpha=0.25, label=proteins[i]+f' (N = {len(data[0])})')
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width()/2
        cy = ry + rectangle.get_height()/2
        ax.annotate(text, (cx, cy), color='black', weight='bold', fontsize=12, ha='center', va='center')
        ax.plot()
    plt.xlabel('Time (s)')
    plt.ylim(0, n)
    ax.yaxis.set_visible(False)
    plt.legend()
    plt.title('Track Timelines (s)')
    plt.show(block=False)

def trinity_exporter(tracks_props, tracks_c2_times, exportpath, proteins):

    print('---------------- Trinity Exporter ----------------')

    # Identify temperatures of proteins of interest (magenta field)
    pattern = r'-\d{1,2}C-'
    proteinref1 = proteins[0]
    proteinref2 = proteins[1]
    tempref_str1, tempref_str2 = '', ''
    for match in re.finditer(pattern, proteinref1):
        span = match.span()
        tempref_str1 = proteinref1[span[0]+1:span[1]-2]
    for match in re.finditer(pattern, proteinref2):
        span = match.span()
        tempref_str2 = proteinref2[span[0]+1:span[1]-2]

    # Check protein C1 and C2 temperature match
    if tempref_str1 and tempref_str2 and tempref_str1 == tempref_str2:

        df_ALL_EXO,  df_ALL_C1, df_ONLY_C1, df_ALL_C2, df_ONLY_C2, df_COLOCALIZED = [pd.DataFrame() for _ in range(6)]
        for key, trck_times in tracks_c2_times.items():

            # Extract protein1 and protein2 temperatures
            protein1 = tracks_props[key]['protein1']
            protein2 = tracks_props[key]['protein2']
            temp_str1 , temp_str2 = '', ''
            for match in re.finditer(pattern, protein1):
                span = match.span()
                temp_str1 = protein1[span[0]+1:span[1]-2]
            for match in re.finditer(pattern, protein2):
                span = match.span()
                temp_str2 = protein2[span[0]+1:span[1]-2]

            if (protein1.replace(temp_str1, tempref_str1) == proteinref1 and protein2.replace(temp_str2, tempref_str2) == proteinref2
                    and tracks_props[key]['ch2_positive'] == 1 and tracks_props[key]['ignore_track'] == 0):

                start_c1 = 0
                lgth_c1 = tracks_props[key]['length']
                end_c1 = tracks_props[key]['length']
                start_c2 = (trck_times[0]-tracks_props[key]['int_preframe'])
                end_c2 = (trck_times[1]-tracks_props[key]['int_preframe'])
                lgth_c2 = trck_times[2]

                ALL_EXO = max(end_c1, end_c2) - min(start_c1, start_c2)
                ALL_C1 = lgth_c1
                ALL_C2 = lgth_c2
                # Assuming C2 is late, that is start after C1 and end after
                ONLY_C1 = max(min(start_c2-start_c1, lgth_c1), 0)
                ONLY_C2 = max(min(end_c2-end_c1, lgth_c2), 0)
                COLOCALIZED = ALL_EXO-ONLY_C1-ONLY_C2
                if temp_str1 in df_ALL_EXO.columns:
                    last_idx = df_ALL_EXO[temp_str1].last_valid_index()
                    df_ALL_EXO.at[last_idx+1, temp_str1] = ALL_EXO
                else:
                    df_ALL_EXO.at[0, temp_str1] = ALL_EXO
                if temp_str1 in df_ALL_C1.columns:
                    last_idx = df_ALL_C1[temp_str1].last_valid_index()
                    df_ALL_C1.at[last_idx+1, temp_str1] = ALL_C1
                else:
                    df_ALL_C1.at[0, temp_str1] = ALL_C1
                if temp_str1 in df_ONLY_C1.columns:
                    last_idx = df_ONLY_C1[temp_str1].last_valid_index()
                    df_ONLY_C1.at[last_idx+1, temp_str1] = ONLY_C1
                else:
                    df_ONLY_C1.at[0, temp_str1] = ONLY_C1
                if temp_str1 in df_ALL_C2.columns:
                    last_idx = df_ALL_C2[temp_str1].last_valid_index()
                    df_ALL_C2.at[last_idx+1, temp_str1] = ALL_C2
                else:
                    df_ALL_C2.at[0, temp_str1] = ALL_C2
                if temp_str1 in df_ONLY_C2.columns:
                    last_idx = df_ONLY_C2[temp_str1].last_valid_index()
                    df_ONLY_C2.at[last_idx + 1, temp_str1] = ONLY_C2
                else:
                    df_ONLY_C2.at[0, temp_str1] = ONLY_C2
                if temp_str1 in df_COLOCALIZED.columns:
                    last_idx = df_COLOCALIZED[temp_str1].last_valid_index()
                    df_COLOCALIZED.at[last_idx + 1, temp_str1] = COLOCALIZED
                else:
                    df_COLOCALIZED.at[0, temp_str1] = COLOCALIZED

        df_ALL_EXO.to_csv(exportpath+'temp_vs_track_S_XXX_ALL_EXOCYST_DURATION.csv', index=False, na_rep='')
        df_ALL_C1.to_csv(exportpath+'temp_vs_track_S_XXX_ALL_C1_DURATION.csv', index=False, na_rep='')
        df_ONLY_C1.to_csv(exportpath+'temp_vs_track_S_XXX_ONLY_C1_DURATION.csv', index=False, na_rep='')
        df_ALL_C2.to_csv(exportpath+'temp_vs_track_S_XXX_ALL_C2_DURATION.csv', index=False, na_rep='')
        df_ONLY_C2.to_csv(exportpath+'temp_vs_track_S_XXX_ONLY_C2_DURATION.csv', index=False, na_rep='')
        df_COLOCALIZED.to_csv(exportpath+'temp_vs_track_S_XXX_COLOCALIZED_DURATION.csv', index=False, na_rep='')
        print('Number of tracks exported for each temperature points:')
        print(df_ALL_EXO.count(axis=0))
        tracks_statistics.call_button.text = f'{df_ALL_EXO.count().sum()} Tracks exported'

    else:
        print('Temperatures of C1 and C2 proteins do not match!')


@magicgui(call_button='Process',
          plot_intensity_profiles={'widget_type': 'Checkbox', 'tooltip': 'Plot C1/C2 intensity profiles (median filtered)'},
          model_C2_track={'widget_type': 'Checkbox', 'tooltip': 'Fit a dual plateau function to C2 intensity profiles'},
          plot_first_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'First C2+ track to plot', 'label': ' '},
          plot_last_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'Last C2+ track to plot', 'label': ' '},
          write_ignore_tracks={'widget_type': 'Checkbox', 'tooltip': 'Flag tracks to ignore in pkl file (Tracks IDs read from .txt file with same name as .pkl file)'})
def curate_tracks(plot_intensity_profiles=True, model_C2_track=False, plot_first_trck=1, plot_last_trck=25, write_ignore_tracks=False):

    # Close all plots
    plt.close('all')

    # Load pkl files from current C1/C2 images
    file_C1 = str(load_images_tiff.imagepath.value).replace('.tif', '.pkl')
    file_C2 = str(load_images_tiff.imagepath2.value).replace('.tif', '.pkl')
    with open(file_C1, 'rb') as file:
        tracks_props = pickle.load(file)
    with open(file_C2, 'rb') as file:
        tracks_c2_times = pickle.load(file)

    # Write ignore track flags from ignore_tracks.txt to C1 pkl file
    if write_ignore_tracks:
        tracks_ignore_file = file_C1.replace('.pkl', '.txt')
        if path.exists(tracks_ignore_file):
            with open(tracks_ignore_file) as file:
                str_values = file.readline().strip().split(',')
                tracks_ignore = [int(v.strip()) for v in str_values if v]
            print(f'Tracks to ignore: {tracks_ignore}')
            cnt = 1
            for key, value in list(tracks_props.items())[:]:
                if tracks_props[key]['ch2_positive'] == 1:
                    if cnt in tracks_ignore:
                        tracks_props[key]['ignore_track'] = 1
                        print(f'Tracks {cnt} set to ignore in pkl file {file_C1}')
                    else:
                        tracks_props[key]['ignore_track'] = 0
                    cnt += 1
            with open(file_C1, 'wb') as file:
                pickle.dump(tracks_props, file)

    if plot_intensity_profiles:
        plot_tracks_intensity(tracks_props, tracks_c2_times, first_trck=plot_first_trck, last_trck=plot_last_trck, int_norm=False)
    if model_C2_track:
        fit_plateaus(tracks_props, first_trck=plot_first_trck, last_trck=plot_last_trck)


@magicgui(call_button='Process',
          groupfiles={'widget_type': 'Checkbox', 'tooltip': 'Process all files from current image folder(s)'},
          plot_average_intensity_profile={'widget_type': 'Checkbox', 'tooltip': 'Plot average intensity profiles for current protein conditions (magenta)'},
          intnorm={'widget_type': 'Checkbox', 'tooltip': 'Normalize intensity'},
          plot_timelines={'widget_type': 'Checkbox', 'tooltip': 'Plot track timelines and compute statistics'},
          export_to_trinity={'widget_type': 'Checkbox', 'tooltip': 'Export to Trinity all available temperatures for current protein conditions (magenta)'})
def tracks_statistics(groupfiles=False, plot_average_intensity_profile=True, intnorm=True, plot_timelines=False, export_to_trinity=False):

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

    if path.exists(files_C1[0]) and path.exists(files_C2[0]):

        # Display file information
        print('---------------- Tracks Statistics ----------------')
        print('File(s) analyzed: ')
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

        if plot_average_intensity_profile:
            plot_tracks_avg_intensity(tracks_props, proteins=proteins, int_norm = intnorm)

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
            cnt = 0
            for key, trck_times in tracks_c2_times.items():
                if tracks_props[key]['ch2_positive'] == 1 and tracks_props[key]['ignore_track'] == 0:
                    ts = tracks_props[key]['frame_timestep']
                    lgths_c1[tracks_props[key]['protein1']].append(tracks_props[key]['length']*ts)
                    starts_c2[tracks_props[key]['protein2']].append(
                        (trck_times[0] - tracks_props[key]['int_preframe'])*ts)
                    ends_c2[tracks_props[key]['protein2']].append((trck_times[1]-tracks_props[key]['int_preframe'])*ts)
                    lgths_c2[tracks_props[key]['protein2']].append(trck_times[2]*ts)
                    cnt += 1
            data_list, proteins = [], []
            print(f'Number of C2 positive C1 tracks used for statistics: {cnt} ({len(tracks_c2_times.items())-cnt} ignored)')
            for protein1 in proteins1:
                if protein1 in lgths_c1:
                    #print(f'{protein1} mean track length: {np.mean(lgths_c1[protein1]):.2f} +/- {np.std(lgths_c1[protein1]):.2f} [range: {np.min(lgths_c1[protein1]):.2f} - {np.max(lgths_c1[protein1]):.2f}]')
                    data_list.append([np.zeros_like(lgths_c1[protein1]), lgths_c1[protein1]])
                    proteins.append(protein1)
            for protein2 in proteins2:
                if protein2 in lgths_c2:
                    #print(f'{protein2} mean track length: {np.mean(lgths_c2[protein2]):.2f} +/- {np.std(lgths_c2[protein2]):.2f} [range: {np.min(lgths_c2[protein2]):.2f} - {np.max(lgths_c2[protein2]):.2f}]')
                    data_list.append([starts_c2[protein2], ends_c2[protein2]])
                    proteins.append(protein2)

            # Call plotting function
            plot_tracks_timelines(data_list, proteins=proteins)

        if export_to_trinity:
            exportpath = path.dirname(load_images_tiff.imagepath.value)+'/Trinity/'
            if not path.exists(exportpath):
                makedirs(exportpath)
            trinity_exporter(tracks_props, tracks_c2_times, exportpath, proteins=proteins)
        else:
            tracks_statistics.call_button.text = 'Process'

    else:
        print('No file to process!')
