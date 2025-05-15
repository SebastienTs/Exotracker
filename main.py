#### Comments on nomenclature and parameters
#
# Contrast: Intensity ratio between spot intensity (1 pixel rad disk) and surrounding (1.5*spot_rad radius disk)
# Max gap: 15 frames can seem high but at this fram rate shorter gaps incur the risk of unlinking reappearing spots
#
#### Results format
#
# The results are saved in two .pkl files (same folder as input images)
#
## C1 pkl file
# 'ch1_int': intensity array (length L)
# 'ch1_ext_int': intensity array (length L+min_startframe+min_afterframe)
# 'ch2_ext_int': intensity array (length L+min_startframe+min_afterframe)
# 'track': panda dataframe (columns: particle, frame, y, x) x L rows
# 'length': L
# 'protein1': name of C1 protein
# 'protein2': name of C2 protein
#
## C2 pkl file
# Numpy array (shape Nx3)
# N rows: one C2 positive track per row
# 3 columns: C2_track_start_frame, C2_track_end_frame, C2_track_length
# C2 tracks start/end/length are estimated from the EXTENDED intensity profiles 'ch2_ext_int'
#
####

import napari
import warnings
from algos import *
from utils import *
from graphs import *
warnings.filterwarnings("ignore", category=FutureWarning)

# Workflow napari widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_images_tiff, area='right', name='Load Images')
dw1.setMinimumHeight(240);dw1.setMaximumHeight(240)
dw2 = viewer.window.add_dock_widget(detect_spots_msdog, area='right', name='Detect Spots')
dw2.setMinimumHeight(140);dw2.setMaximumHeight(140)
dw3 = viewer.window.add_dock_widget(track_spots_trackpy, area='right', name='Track Spots')
dw3.setMinimumHeight(250);dw3.setMaximumHeight(250)
dw4 = viewer.window.add_dock_widget(analyze_tracks_int_gate, area='right', name='Analyze Tracks')
dw4.setMinimumWidth(360)
dw5 = viewer.window.add_dock_widget(graph_tracks, area='right', name='Analyze Data')
dw5.setMinimumWidth(360)

# Display napari viewer
napari.run()
