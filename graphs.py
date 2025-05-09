import math
import pickle
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from settings import frame_timestep, proteins
from magicgui import magicgui
from pathlib import Path
from utils import *
warnings.filterwarnings("ignore")

# Configuration
groupfiles_def = False        # When grouping files the C1 and C2 dictionaries stored in all pkl files are merged
int_profiles_def = False      # Display all intensity time plots
avgint_profiles_def = True    # Display average intensity time plots (C1 tracks normalized)
timelines_def  = True         # Display average timelines (C1 tracks normalized)
mx_trck_def = 999             # Maximum number of tracks to include in the statistics
mx_prefrc_def = 0             # Maximum relative time before C1 tracks normalized to C1 track length (plot only)
mx_postfrc_def = 0.5          # Maximum relative time after C1 tracks normalized to C1 track length (plot only)

@magicgui(call_button='Analyze',
          groupfiles={'widget_type': 'Checkbox', 'tooltip': 'Process all .pkl files from current image folder instead of only current image'},
          int_profiles={'widget_type': 'Checkbox', 'tooltip': 'Plot intensity profiles'},
          avgint_profiles={'widget_type': 'Checkbox', 'tooltip': 'Plot average intensity profile'},
          timelines={'widget_type': 'Checkbox', 'tooltip': 'Plot timelines'},
          mx_trck={'widget_type': 'IntSlider', 'min': 1, 'max': 999, 'tooltip': 'Maximum number of positive tracks to analyze'},
          mx_prefrc={'widget_type': 'FloatSlider', 'min': 0, 'max': 1, 'tooltip': 'Fraction of'},
          mx_postfrc={'widget_type': 'FloatSlider', 'min': 0, 'max': 1, 'tooltip': 'Fraction of'})
def graph_tracks(groupfiles=groupfiles_def, int_profiles=int_profiles_def, avgint_profiles=avgint_profiles_def,
                 timelines=timelines_def, mx_trck=mx_trck_def, mx_prefrc=mx_prefrc_def, mx_postfrc=mx_postfrc_def):

    reanalyze = False   # Recompute C2 tracks from intensity gating (debug, bypass posterior plots and analysis)
    model = False       # Estimate C2 intensity plateaus by fitting logistic functions

    # Result files have same name as tif files but .pkl extension
    results_C1 = load_images_tiff.imagepath.value
    results_C2 = load_images_tiff.imagepath2.value

    # All pkl files in the folder of analyzed file are grouped
    extension = Path(results_C1).suffix
    newextension = '.pkl'
    results_C1 = str(results_C1).replace(extension, newextension)
    results_C2 = str(results_C2).replace(extension, newextension)
    if groupfiles:
        folder = Path(results_C1).parent
        files_C1 = [str(f) for f in folder.glob(f'*{newextension}') if f.is_file() and '_C1' in f.name]
        files_C2 = [str(f) for f in folder.glob(f'*{newextension}') if f.is_file() and '_C2' in f.name]
    else:
        files_C1 = [results_C1]
        files_C2 = [results_C2]

    # Display analysis information
    print(f'Frame timestep: {frame_timestep} s')
    print('Analyzing this .pkl files: ')
    print([Path(file_C1).name for file_C1 in files_C1])
    print([Path(file_C2).name for file_C2 in files_C2])

    # Add entries of each pkl file to common dictionaries
    tracks_props = {}
    tracks_c2_times = {}
    for i, (file_C1, file_C2) in enumerate(zip(files_C1, files_C2)):
        with open(file_C1, 'rb') as file:
            tracks_props.update({str(k) + '_'+str(i): v for k, v in pickle.load(file).items()})
        with open(file_C2, 'rb') as file:
            tracks_c2_times.update({str(k) + '_'+str(i): v for k, v in pickle.load(file).items()})

    # Re-analyze C2 tracks
    if reanalyze:
        analyze_and_plot_C2_tracks_intensity(tracks_props, trckperplot = 3, medrad = 9, trck_thr = 0.3)
    elif model:
        model_C2_tracks_intensity(tracks_props, timestep=frame_timestep)
    else:
        # Initialization
        starts_c1, starts_c2 = [], []
        stops_c1, stops_c2 = [], []
        lgths_c1, lgths_c2 = [], []
        # Loop over all C2 tracks (only C2 colocalized C1 tracks considered)
        for key, trck_times in tracks_c2_times.items():
            lgths_c1.append(tracks_props[key]['length'])
            starts_c2.append(trck_times[0])
            stops_c2.append(trck_times[1])
            lgths_c2.append(trck_times[2])
        print(f'C1 mean track length: {np.mean(lgths_c1)*frame_timestep:.2f} +/- {np.std(lgths_c1)*frame_timestep:.2f} [range: {np.min(lgths_c1)*frame_timestep} - {np.max(lgths_c1)*frame_timestep}]')
        print(f'C2 mean track length: {np.mean(lgths_c2)*frame_timestep:.2f} +/- {np.std(lgths_c2)*frame_timestep:.2f} [range: {np.min(lgths_c2)*frame_timestep} - {np.max(lgths_c2)*frame_timestep}]')
        data_list = [[np.zeros_like(lgths_c1), lgths_c1], [starts_c2, stops_c2]]
        if int_profiles:
            plot_C1_C2_tracks_intensity(tracks_props, tracks_c2_times, mx_trck = mx_trck, medrad = 6, int_norm = True)
        if avgint_profiles:
            plot_C1_C2_tracks_avg_intensity(tracks_props, mx_trck = mx_trck, mx_prefrc = mx_prefrc, mx_postfrc=mx_postfrc, medrad = 6, int_norm = True, proteins=proteins)
            plt.title('Average intensity profile (N='+str(len(lgths_c1))+')')
        if timelines:
            plot_timelines(data_list, proteins=proteins, mx_frame=250, timestep=frame_timestep)
            plt.title('C1 and C2 track timelines (N='+str(len(lgths_c1))+')')
        #plt.show(block = True)
