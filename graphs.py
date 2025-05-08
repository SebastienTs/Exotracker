import math
import pickle
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from utils import *
from settings import frame_timestep, filename, filename2, proteins
warnings.filterwarnings("ignore")

# Configuration
reanalyze = False               # Recompute C2 tracks from intensity gating (debug, bypass posterior plots and analysis)
model = False                   # Estimate C2 intensity plateaus by fitting logistic functions
intensity_profiles = False      # Display all intensity time plots
avg_intensity_profile = True    # Display average intensity time plots (C1 tracks normalized)
timelines = True                # Display average timelines (C1 tracks normalized)
mx_trck = 9999                  # Maximum number of tracks to include in the statistics
mx_prefrc = 0                   # Maximum relative time before C1 tracks normalized to C1 track length (plot only)
mx_postfrc = 0.5                # Maximum relative time after C1 tracks normalized to C1 track length (plot only)

# Load results files (add loop to combine)
results_c1 = filename.replace('.tif', '.pkl')
results_c2 = filename2.replace('.tif', '.pkl')
with open(results_c1, 'rb') as file:
    tracks_props = pickle.load(file)
with open(results_c2, 'rb') as file:
    tracks_c2_times = pickle.load(file)

# Re-analyze C2 tracks
print(f'Frame timestep: {frame_timestep} s')
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
    print(f'C1 mean track length: {np.mean(lgths_c1)*frame_timestep:.2f} +/- {np.std(lgths_c1)*frame_timestep:.2f} (range: {np.min(lgths_c1)} - {np.max(lgths_c1)}, N: {len(lgths_c1)})')
    print(f'C2 mean track length: {np.mean(lgths_c2)*frame_timestep:.2f} +/- {np.std(lgths_c2)*frame_timestep:.2f} (range: {np.min(lgths_c2)} - {np.max(lgths_c2)}, N: {len(lgths_c2)})')
    data_list = [[np.zeros_like(lgths_c1), lgths_c1], [starts_c2, stops_c2]]
    if intensity_profiles:
        plot_C1_C2_tracks_intensity(tracks_props, tracks_c2_times, mx_trck = mx_trck, medrad = 6, int_norm = True)
    if avg_intensity_profile:
        plot_C1_C2_tracks_avg_intensity(tracks_props, mx_trck = mx_trck, mx_prefrc = mx_prefrc, mx_postfrc=mx_postfrc, medrad = 6, int_norm = True)
    if timelines:
        plot_timelines(data_list, proteins=proteins, mx_frame=250, timestep=frame_timestep)
    plt.show(block = True)
