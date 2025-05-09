#### Results format
# The results are saved as two .pkl files in same folder as the C1/C2 input images
## C1 file
# 'ch1_int': array of length L
# 'ch1_ext_int': array of length L+c2_preframes+c2_postframes
# 'ch2_ext_int': array of length L+c2_preframes+c2_postframes
# 'track': dataframe (particle / frame / y / x), L rows
# 'length': L
## C2 file
# Array of size N x 3 (N is the number of C2 positive tracks)
# One track per row: C2_track_start_frame, C2_track_end_frame, C2_track_length
# C2 tracks are estimated from the intensity profiles ch2_ext_int

import napari
import warnings
from algos import *
from utils import *
from graphs import *
warnings.filterwarnings("ignore", category=FutureWarning)

# Workflow configuration
image_loader = load_images_tiff
detect_spots = detect_spots_msdog
track_spots = track_spots_trackpy
analyze_tracks = analyze_tracks_int_gate
graph_data = graph_tracks

# Workflow runner (with default settings)
def run_workflow():
    image_loader()
    detect_spots()
    track_spots()
    analyze_tracks()

# Workflow napari widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(image_loader, area='right', name='Load Images')
dw1.setMinimumHeight(200);dw1.setMaximumHeight(200)
dw2 = viewer.window.add_dock_widget(detect_spots, area='right', name='Detect Spots')
dw2.setMinimumHeight(160);dw2.setMaximumHeight(160)
dw3 = viewer.window.add_dock_widget(track_spots, area='right', name='Track Spots')
dw3.setMinimumHeight(240);dw3.setMaximumHeight(240)
dw4 = viewer.window.add_dock_widget(analyze_tracks, area='right', name='Analyze Tracks')
dw4.setMinimumWidth(360)
dw5 = viewer.window.add_dock_widget(graph_data, area='right', name='Analyze Data')
dw5.setMinimumWidth(360)

# Run workflow
#run_workflow()

# Display napari viewer
napari.run()
