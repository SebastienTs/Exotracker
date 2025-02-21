import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from utils import *
import warnings
warnings.filterwarnings("ignore")

# Load results files (add loop to combine)
with open('D:/Projects/UPF/Live_exo84_sec9/Oblique/30C/5_Montage/Montage_preprocessed_C1.pkl', 'rb') as file:
    tracks_props = pickle.load(file)
with open('D:/Projects/UPF/Live_exo84_sec9/Oblique/30C/5_Montage/Montage_preprocessed_C2.pkl', 'rb') as file:
    tracks2_times = pickle.load(file)

# Re-analyze C2 tracks
#analyze_and_plot_C2_tracks(tracks_props, 400, 250, 3, 9, 0.44)

lgths = []
for trck_time in tracks2_times.values():
    lgths.append(trck_time[2])
print(np.mean(lgths), np.std(lgths), np.min(lgths), np.max(lgths))
