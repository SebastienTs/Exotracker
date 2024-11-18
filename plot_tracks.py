import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def tile_windows():
    figs = plt.get_fignums()
    n = len(figs)
    cols = math.ceil(math.sqrt(n))
    for i, num in enumerate(figs):
        plt.figure(num)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50 + i % cols * 250, 50 + i // cols * 200, 250, 175)


def longest_non_zero_sequence(seq):
    st, en = max(((i, i + len(s)) for i, s in enumerate(''.join(map(str, seq)).split('0')) if s),
                     key=lambda x: x[1] - x[0], default=(0, 0))
    return st, en, en - st


with open('D:/Projects/UPF/TIRF/30C/5_Montage/Montage_preprocessed_C1.pkl', 'rb') as file:
    tracks_kept_props = pickle.load(file)

cnt = 1
plt.figure()
for key, value in list(tracks_kept_props.items())[:]:
    if tracks_kept_props[key]['ch2_positive'] == 1:
        vals = np.array(tracks_kept_props[key]['ch2_int'])
        vals_flt = medfilt(vals, kernel_size=9)
        vals_flt = (vals_flt - vals_flt.min()) / (vals_flt.max() - vals_flt.min())
        trackflag = (vals_flt >= 0.4).astype(int)
        trackflag = medfilt(trackflag, kernel_size=5)
        start, end , tracklgth = longest_non_zero_sequence(trackflag)
        print(start, end, tracklgth)
        plt.plot(vals_flt)
        if cnt == 4:
            plt.figure()
            cnt = 0
        cnt += 1

tile_windows()
plt.show(block=True)
