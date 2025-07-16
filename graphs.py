import re
import pickle
import numpy as np
import pandas as pd
import platform, shutil, subprocess
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from magicgui import magicgui
from pathlib import Path
from collections import defaultdict
from os import path, makedirs
from algos import *
from settings import mx_prefrc, mx_postfrc
warnings.filterwarnings("ignore")

## Fit a linear combination of logistic functions to C2 intensity profiles
def fit_plateaus(tracks_props, first_trck, last_trck, nplateaus, steepness):

    # Radius of rolling median filter used for C2 track detection
    medrad = analyze_tracks_int_gate.track_c2_int_medrad.value

    # Plot loop
    mpl.rcParams['toolbar'] = 'None'
    cnt = 1
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['ch2_positive'] == 1:
            if first_trck <= cnt <= last_trck and tracks_props[key]['ignore_track'] == 0:
                int_c2 = tracks_props[key]['ch2_ext_int']
                int_c2 = medfilt(int_c2, kernel_size=2*medrad+1)
                int_c2 = (int_c2-min(int_c2))/(max(int_c2)-min(int_c2)).astype(np.float64)
                x = np.arange(0, len(int_c2)).astype(np.float64)
                model = make_logistic_combination(n=(2*nplateaus), steepness=steepness)
                p0 = np.concatenate([np.tile([0.5, -0.5], nplateaus), np.linspace(0, len(int_c2), num=(2*nplateaus))])
                lbounds =  np.concatenate([np.tile([0, -1], nplateaus), np.zeros(2*nplateaus)])
                rbounds = np.concatenate([np.tile([1, 0], nplateaus), len(int_c2)*np.ones(2*nplateaus)])
                try:
                    popt, _ = curve_fit(model, x, int_c2, maxfev=5000, p0=p0, bounds=(lbounds, rbounds), method='trf')
                except RuntimeError:
                     popt = np.array(zeros_like(int_c2))
                plt.figure()
                plt.plot(x, int_c2, label='Data')
                plt.plot(x, model(x, *popt), 'r-', label='Fit')
                plt.xlim(0, len(int_c2))
                plt.title(f'C2 Track {cnt}')
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)


# Plot C1-C2 track pairs intensity profiles (rolling median filtered)
def  plot_tracks_intensity(tracks_props, tracks_c2_times, first_trck, last_trck, int_norm):

    # Radius of rolling median filter used for C2 track detection
    medrad = analyze_tracks_int_gate.track_c2_int_medrad.value

    # Plot loop
    mpl.rcParams['toolbar'] = 'None'
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
                start, end, lgth = tracks_c2_times[key][0], tracks_c2_times[key][1], tracks_c2_times[key][2]
                int_c2[1:start], int_c2[end:] = np.NaN, np.NaN
                plt.plot(int_c2, color='green')
                plt.grid()
                xstart, ystart, fstart = track.iloc[0]['x'], track.iloc[0]['y'], track.iloc[0]['frame']
                plt.title(f'C2 Track {cnt} (T: {int(fstart-1)}, Y: {int(ystart)}, X: {int(xstart)})')
            cnt += 1
    tile_windows(300, 200)
    plt.show(block=False)


# Plot C1/C2 average intensity profiles
def plot_tracks_avg_intensity(tracks_props, exp_proteins, int_norm):

    # C1 intensity profile is resampled to fixed size rsplgth
    # C2 buffer is 4*rsplgth to accomodate:
    #   - C2 intensity profile (resampled C1 length)
    #   - pre-analysis (up to resampled C1 length)
    #   - post-analysis (up to resampled C1 length)
    #   - pre-analysis alignment (up to resampled C1 length)
    rsplgth = 512
    arr_int_c1 = np.full((int(rsplgth), len(tracks_props)), np.nan)
    arr_int_c2 = np.full((int(4*rsplgth), len(tracks_props)), np.nan)

    # Plot loop
    mpl.rcParams['toolbar'] = 'toolbar2'
    cnt, cnt2 = 0, 0
    for key, value in list(tracks_props.items())[:]:
        if tracks_props[key]['protein1'] == exp_proteins[0] and tracks_props[key]['protein2'] == exp_proteins[1] and tracks_props[key]['ch2_positive'] == 1:
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
                else:
                    int_c1 = (int_c1-min(int_c1))
                    int_c2 = (int_c2-min(int_c2))
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

    # Compute relative maxima shift
    relmaxshft = (np.argmax(avg_int_c2)-np.argmax(avg_int_c1))/rsplgth-1

    # Plot
    fig = plt.figure()
    plt.plot(np.arange(0, 1, 1/rsplgth), avg_int_c1, color='red', label=exp_proteins[0])
    if int_norm:
        fig.gca().fill_between(np.arange(0, 1, 1 / rsplgth), avg_int_c1 - std_int_c1, avg_int_c1 + std_int_c1, color='red', alpha=0.2)
    plt.plot(np.arange(-1, 3, 1/rsplgth), avg_int_c2, color='green', label=exp_proteins[1])
    if int_norm:
        fig.gca().fill_between(np.arange(-1, 3, 1/rsplgth), avg_int_c2-std_int_c2, avg_int_c2+std_int_c2, color='green', alpha=0.2)
        plt.title(f'Average normalized tracks intensity (\u0394: {relmaxshft:.2f}, N=' + str(cnt) + ')')
        plt.xlabel('C1 track normalized time')
        plt.ylabel('AU')
    else:
        plt.title(f'Average tracks intensity (\u0394: {relmaxshft:.2f}, N=' + str(cnt) + ')')
        plt.xlabel('C1 track normalized time')
        plt.ylabel('Intensity (minimum subtracted)')
    plt.xlim(-mx_prefrc, 1+mx_postfrc)
    plt.legend()
    plt.show(block=False)


# Plot proteins timelines (start/end/length)
def plot_tracks_timelines(data_list, exp_proteins, std_and_delta, time_norm_c1):
    cols = ['red', 'green', 'blue', 'orange', 'pink', 'cyan', 'yellow', 'gray']
    n = len(data_list)
    figsize = (6,3*n)
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    combined = sorted(zip(exp_proteins, data_list), key=lambda x: x[0])
    exp_proteins, data_list = zip(*combined)

    # Plot loop
    mpl.rcParams['toolbar'] = 'toolbar2'
    for i, data in enumerate(data_list):
        start = np.mean(data[0])
        end = np.mean(data[1])
        lgths = np.array(data[1])-np.array(data[0])
        if 'exo84' in exp_proteins[i]:
            if time_norm_c1:
                start = 0
                end = 1
            startref = start
            endref = end
            lengthref = np.mean(lgths)
            if std_and_delta:
                text = f'{np.mean(lgths):.1f} \n \u0394={100*np.std(lgths)/np.mean(lgths):.0f}%'
            else:
                text = f'{np.mean(lgths):.1f}'
            height = 1
        else:
            if time_norm_c1:
                start = start/lengthref
                end = end/lengthref
            if std_and_delta:
                if time_norm_c1:
                    text = f'{np.mean(lgths):.1f} \n {(start-startref):+.2f}, \u0394={100 * np.std(lgths)/np.mean(lgths):.0f}%, {(end-endref):+.2f}'
                else:
                    text = f'{np.mean(lgths):.1f} \n {(start-startref)/lengthref:+.2f}, \u0394={100*np.std(lgths)/np.mean(lgths):.0f}%, {(end-endref)/lengthref:+.2f}'
            else:
                text = f'{np.mean(lgths):.1f}'
            height = 0.75
        # Plot timelines
        rectangle = Rectangle((start, n-1-i+1-height), end-start, height, facecolor=cols[i%8], alpha=0.25, label=exp_proteins[i] + f' (N = {len(data[0])}')
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
    plt.title('Tracks Timeline')
    plt.show(block=False)


# Export temperature grouped track analysis results to Trinity format
def trinity_exporter(tracks_props, tracks_c2_times, exportpath, exp_proteins):

    print('---------------- Trinity Exporter ----------------')

    # Identify temperatures of proteins of interest (magenta field)
    proteinref1 = exp_proteins[0]
    proteinref2 = exp_proteins[1]
    proteinname1 = proteinref1.split('-')[-1]
    proteinname2 = proteinref2.split('-')[-1]
    tempref_str1, tempref_str2 = '', ''
    pattern = r'-\d{1,2}C-'
    for match in re.finditer(pattern, proteinref1):
        span = match.span()
        tempref_str1 = proteinref1[span[0]+1:span[1]-2]
    for match in re.finditer(pattern, proteinref2):
        span = match.span()
        tempref_str2 = proteinref2[span[0]+1:span[1]-2]

    # Check that temperature for proteins C1 and C2 match
    if tempref_str1 and tempref_str2 and tempref_str1 == tempref_str2:

        # Create dataframes to store results
        df_all_exo,  df_all_c1, df_only_c1, df_all_c2, df_only_c2, df_colocalized = [pd.DataFrame() for _ in range(6)]

        for key, trck_times in tracks_c2_times.items():

            # Extract protein1 and protein2 temperatures
            protein1 = tracks_props[key]['protein1']
            protein2 = tracks_props[key]['protein2']
            temp_str1 , temp_str2 = '', ''
            pattern = r'-\d{1,2}C-'
            for match in re.finditer(pattern, protein1):
                span = match.span()
                temp_str1 = protein1[span[0]+1:span[1]-2]
            for match in re.finditer(pattern, protein2):
                span = match.span()
                temp_str2 = protein2[span[0]+1:span[1]-2]

            # Ensure valid temperatures were found and proteins have matching temperatures for both channels
            if (protein1.replace(temp_str1, tempref_str1) == proteinref1 and protein2.replace(temp_str2, tempref_str2) == proteinref2
                    and tracks_props[key]['ch2_positive'] == 1 and tracks_props[key]['ignore_track'] == 0):

                # Extract C1 and C2 tracks information
                start_c1 = 0
                lgth_c1 = tracks_props[key]['length']*tracks_props[key]['frame_timestep']
                end_c1 = tracks_props[key]['length']*tracks_props[key]['frame_timestep']
                start_c2 = (trck_times[0]-tracks_props[key]['int_preframe'])*tracks_props[key]['frame_timestep']
                end_c2 = (trck_times[1]-tracks_props[key]['int_preframe'])*tracks_props[key]['frame_timestep']
                lgth_c2 = trck_times[2]*tracks_props[key]['frame_timestep']

                # Duration of all exocytosis (at least C1 or C2 protein)
                all_exo = max(end_c1, end_c2)-min(start_c1, start_c2)

                # Duration of C1 and C2 proteins
                all_c1 = lgth_c1
                all_c2 = lgth_c2

                # C1 and C2 overlap (colocalized)
                colocalized = max(0, min(end_c1, end_c2)-max(start_c1, start_c2))

                # Assuming C2 is late, that is it starts after C1 and end after it too
                only_c1 = lgth_c1-colocalized
                only_c2 = lgth_c2-colocalized

                # Append results to dataframes
                if temp_str1 in df_all_exo.columns:
                    last_idx = df_all_exo[temp_str1].last_valid_index()
                    df_all_exo.at[last_idx+1, temp_str1] = all_exo
                else:
                    df_all_exo.at[0, temp_str1] = all_exo
                if temp_str1 in df_all_c1.columns:
                    last_idx = df_all_c1[temp_str1].last_valid_index()
                    df_all_c1.at[last_idx+1, temp_str1] = all_c1
                else:
                    df_all_c1.at[0, temp_str1] = all_c1
                if temp_str1 in df_only_c1.columns:
                    last_idx = df_only_c1[temp_str1].last_valid_index()
                    df_only_c1.at[last_idx+1, temp_str1] = only_c1
                else:
                    df_only_c1.at[0, temp_str1] = only_c1
                if temp_str1 in df_all_c2.columns:
                    last_idx = df_all_c2[temp_str1].last_valid_index()
                    df_all_c2.at[last_idx+1, temp_str1] = all_c2
                else:
                    df_all_c2.at[0, temp_str1] = all_c2
                if temp_str1 in df_only_c2.columns:
                    last_idx = df_only_c2[temp_str1].last_valid_index()
                    df_only_c2.at[last_idx + 1, temp_str1] = only_c2
                else:
                    df_only_c2.at[0, temp_str1] = only_c2
                if temp_str1 in df_colocalized.columns:
                    last_idx = df_colocalized[temp_str1].last_valid_index()
                    df_colocalized.at[last_idx + 1, temp_str1] = colocalized
                else:
                    df_colocalized.at[0, temp_str1] = colocalized

        # Reorder the columns of the dataframe in ascending temperature order
        df_all_exo = df_all_exo.iloc[:, np.argsort(df_all_exo.columns.tolist())]
        df_all_c1 = df_all_c1.iloc[:, np.argsort(df_all_c1.columns.tolist())]
        df_only_c1 = df_only_c1.iloc[:, np.argsort(df_only_c1.columns.tolist())]
        df_all_c2 = df_all_c2.iloc[:, np.argsort(df_all_c2.columns.tolist())]
        df_only_c2 = df_only_c2.iloc[:, np.argsort(df_only_c2.columns.tolist())]
        df_colocalized = df_colocalized.iloc[:, np.argsort(df_colocalized.columns.tolist())]

        # Save the dataframes as CSV files
        df_all_exo.to_csv(exportpath+'temp_vs_track_S_XXX_ALL_EXOCYST_DURATION.csv', index=False, na_rep='')
        df_all_c1.to_csv(exportpath+f'temp_vs_track_S_XXX_ALL_{proteinname1}_DURATION.csv', index=False, na_rep='')
        df_only_c1.to_csv(exportpath+f'temp_vs_track_S_XXX_ONLY_{proteinname1}_DURATION.csv', index=False, na_rep='')
        df_all_c2.to_csv(exportpath+f'temp_vs_track_S_XXX_ALL_{proteinname2}_DURATION.csv', index=False, na_rep='')
        df_only_c2.to_csv(exportpath+f'temp_vs_track_S_XXX_ONLY_{proteinname2}_DURATION.csv', index=False, na_rep='')
        df_colocalized.to_csv(exportpath+'temp_vs_track_S_XXX_COLOCALIZED_DURATION.csv', index=False, na_rep='')
        print('Number of tracks exported for each temperature points:')
        print(df_all_exo.count(axis=0))
        tracks_statistics.call_button.text = f'{df_all_exo.count().sum()} Tracks exported'

    else:
        print('Incorrect temperatures or temperatures not matching for both channels, skipping this condition!')


@magicgui(call_button='Process',
          plot_intensity_profiles={'widget_type': 'Checkbox', 'tooltip': 'Plot C1/C2 intensity profiles (median filtered)'},
          plot_first_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'First C2+ track to plot', 'label': ' '},
          plot_last_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 250, 'tooltip': 'Last C2+ track to plot', 'label': ' '},
          model_c2_track={'widget_type': 'Checkbox', 'tooltip': 'Fit a dual plateau function to C2 intensity profiles'},
          nplateaus = {'widget_type': 'IntSlider', 'min': 1, 'max': 5, 'tooltip': 'Number of plateaus used for the function model', 'label': 'Nb. plateaus'},
          steepness = {'widget_type': 'FloatSlider', 'min': 0.05, 'max': 1, 'tooltip': 'Steepness of the plateaus'},
          write_ignore_tracks={'widget_type': 'Checkbox', 'label': 'Update/Write ignore tracks', 'tooltip': 'Flag tracks to ignore in pkl file (Tracks IDs read from .txt file with same name as .pkl file)'})
def curate_tracks(plot_intensity_profiles=True, plot_first_trck=1, plot_last_trck=25, model_c2_track=False,
                  nplateaus=2, steepness=0.75, write_ignore_tracks=False):

    # Close all plots
    plt.close('all')

    # Load pkl files from current C1/C2 images
    file_c1 = str(load_images_tiff.imagepath.value).replace('.tif', '.pkl')
    file_c2 = str(load_images_tiff.imagepath2.value).replace('.tif', '.pkl')

    if path.exists(file_c1) and path.exists(file_c2):
        with open(file_c1, 'rb') as file:
            tracks_props = pickle.load(file)
        with open(file_c2, 'rb') as file:
            tracks_c2_times = pickle.load(file)

        # Write ignore track flags from ignore_tracks.txt to C1 pkl file
        if write_ignore_tracks:
            tracks_ignore_file = file_c1.replace('.pkl', '.txt')
            # Create an empty file if ignore file doesn't exist
            if not path.exists(tracks_ignore_file):
                with open(tracks_ignore_file, "w"):
                    pass
            with open(tracks_ignore_file) as file:
                # Open ignore file in text editor and wait for user to close it
                if platform.system() == "Windows":
                    subprocess.call(['notepad', tracks_ignore_file])
                elif platform.system() == "Linux":
                    if shutil.which("gedit"):
                        subprocess.call(['gedit', tracks_ignore_file])
                    elif shutil.which("kwrite"):
                        subprocess.call(['kwrite', tracks_ignore_file])
                elif platform.system() == "Darwin":
                    subprocess.call(['textedit', tracks_ignore_file])
                str_values = file.readline().strip().split(',')
                tracks_ignore = [int(v.strip()) for v in str_values if v]
            print(f'Tracks to ignore: {tracks_ignore}')
            cnt = 1
            for key, value in list(tracks_props.items())[:]:
                if tracks_props[key]['ch2_positive'] == 1:
                    if cnt in tracks_ignore:
                        tracks_props[key]['ignore_track'] = 1
                        print(f'Tracks {cnt} set to ignore in pkl file {file_c1}')
                    else:
                        tracks_props[key]['ignore_track'] = 0
                    cnt += 1
            with open(file_c1, 'wb') as file:
                pickle.dump(tracks_props, file)

        if plot_intensity_profiles:
            plot_tracks_intensity(tracks_props, tracks_c2_times, first_trck=plot_first_trck,
                                  last_trck=plot_last_trck, int_norm=False)
        if model_c2_track:
            fit_plateaus(tracks_props, first_trck=plot_first_trck, last_trck=plot_last_trck, nplateaus=nplateaus,
                         steepness=steepness)
    else:
        print('The results files (.pkl) do not exist for the selected images!')


@magicgui(call_button='Process',
          groupfiles={'widget_type': 'Checkbox', 'tooltip': 'Process all files from C1/C2 images folder(s)'},
          plot_average_intensity_profile={'widget_type': 'Checkbox', 'tooltip': 'Plot average intensity profiles for current protein conditions'},
          intnorm={'widget_type': 'Checkbox', 'tooltip': 'Normalize intensity'},
          plot_timelines={'widget_type': 'Checkbox', 'tooltip': 'Plot track timelines and compute statistics'},
          std_and_delta={'widget_type': 'Checkbox', 'tooltip': 'Display standard deviation and C1/C2 start/end relative delays'},
          time_norm_c1={'widget_type': 'Checkbox', 'tooltip': 'Normalize all C1 durations'},
          export_to_trinity={'widget_type': 'Checkbox', 'tooltip': 'Export to Trinity all available temperatures for current protein conditions'})
def tracks_statistics(groupfiles=False, plot_average_intensity_profile=True, intnorm=True, plot_timelines=False,
                      std_and_delta=False, time_norm_c1=False, export_to_trinity=False):

    # Proteins of the current dataset
    proteins_str = load_images_tiff.proteins.value
    stripped = proteins_str[1:-1].replace("'", "")
    exp_proteins = [x.strip() for x in stripped.split(',')]

    # Results files: same names as input images but .pkl extension
    files_c1 = [str(load_images_tiff.imagepath.value).replace('.tif', '.pkl')]

    # Retrieve all results files in current C1/C2 image folder(s)
    if groupfiles:
        files_c1 = [str(f) for f in Path(files_c1[0]).parent.glob(f'*{".pkl"}') if f.is_file() and '_C1' in f.name]

    if path.exists(files_c1[0]):

        # Display file information
        print('---------------- Tracks Statistics ----------------')
        print('File(s) analyzed: ')
        print([Path(file_C1).name for file_C1 in files_c1])

        # Aggregate content from pkl files to dictionaries
        tracks_props = {}
        tracks_c2_times = {}
        for i, file_C1 in enumerate(files_c1):
            with open(file_C1, 'rb') as file:
                tracks_props.update({str(k) + '_'+str(i): v for k, v in pickle.load(file).items()})
            file_c2 = file_C1.replace('_C1','_C2')
            with open(file_c2, 'rb') as file:
                tracks_c2_times.update({str(k) + '_'+str(i): v for k, v in pickle.load(file).items()})

        if plot_average_intensity_profile:
            plot_tracks_avg_intensity(tracks_props, exp_proteins=exp_proteins, int_norm = intnorm)

        if export_to_trinity:
            exportpath = path.dirname(load_images_tiff.imagepath.value)+'/Trinity/'
            if not path.exists(exportpath):
                makedirs(exportpath)
            trinity_exporter(tracks_props, tracks_c2_times, exportpath, exp_proteins=exp_proteins)
        else:
            tracks_statistics.call_button.text = 'Process'

        if plot_timelines:
            # Scan for all proteins present in pkl files of current image
            proteins1 = set()
            proteins2 = set()
            for key in tracks_props.keys():
                if tracks_props[key]['ch2_positive'] == 1 and tracks_props[key]['ignore_track'] == 0:
                    proteins1.add(tracks_props[key]['protein1'])
                    proteins2.add(tracks_props[key]['protein2'])
            print(f'Analyzing timelines from proteins:')
            print(proteins1, proteins2)

            # Gather measurements
            lgths_c1 = defaultdict(list)
            starts_c2 = defaultdict(list)
            ends_c2 = defaultdict(list)
            lgths_c2 = defaultdict(list)
            cnt = 0
            for key, trck_times in tracks_c2_times.items():
                if tracks_props[key]['ch2_positive'] == 1 and tracks_props[key]['ignore_track'] == 0:
                    ts = tracks_props[key]['frame_timestep']
                    lgths_c1[tracks_props[key]['protein1']].append(tracks_props[key]['length']*ts)
                    starts_c2[tracks_props[key]['protein2']].append((trck_times[0]-tracks_props[key]['int_preframe'])*ts)
                    ends_c2[tracks_props[key]['protein2']].append((trck_times[1]-tracks_props[key]['int_preframe'])*ts)
                    lgths_c2[tracks_props[key]['protein2']].append(trck_times[2]*ts)
                    cnt += 1
            data_list, exp_proteins = [], []
            print(f'Number of C2 positive C1 tracks used for statistics: {cnt} ({len(tracks_c2_times.items())-cnt} ignored)')
            
            for protein1 in proteins1:
                if protein1 in lgths_c1:
                    #print(f'{protein1} mean track length: {np.mean(lgths_c1[protein1]):.2f} +/- {np.std(lgths_c1[protein1]):.2f} [range: {np.min(lgths_c1[protein1]):.2f} - {np.max(lgths_c1[protein1]):.2f}]')
                    data_list.append([np.zeros_like(lgths_c1[protein1]), lgths_c1[protein1]])
                    exp_proteins.append(protein1)
            for protein2 in proteins2:
                if protein2 in lgths_c2:
                    #print(f'{protein2} mean track length: {np.mean(lgths_c2[protein2]):.2f} +/- {np.std(lgths_c2[protein2]):.2f} [range: {np.min(lgths_c2[protein2]):.2f} - {np.max(lgths_c2[protein2]):.2f}]')
                    data_list.append([starts_c2[protein2], ends_c2[protein2]])
                    exp_proteins.append(protein2)

            # Call plotting function
            plot_tracks_timelines(data_list, exp_proteins=exp_proteins, std_and_delta=std_and_delta, time_norm_c1=time_norm_c1)

    else:
        print('The results files (.pkl) do not exist for the selected images!')
