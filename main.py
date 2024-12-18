import napari
from algos import *
from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Workflow configuration
image_loader = load_image_tiff
detect_spots = detect_spots_msdog
track_spots = track_spots_pytrack

# Worflow widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(image_loader, area='right', name='Load Images')
dw1.setMinimumHeight(200);dw1.setMaximumHeight(200)
dw2 = viewer.window.add_dock_widget(detect_spots, area='right', name='Detect Spots')
dw2.setMinimumHeight(160);dw2.setMaximumHeight(160)
dw3 = viewer.window.add_dock_widget(track_spots, area='right', name='Track Spots')
dw3.setMinimumHeight(240);dw3.setMaximumHeight(240)
dw4 = viewer.window.add_dock_widget(analyze_tracks, area='right', name='Analyze Tracks')
dw4.setMinimumWidth(360)

# Trigger workflow and display viewer
#image_loader()
#detect_spots()
#track_spots()
#filter_tracks()
napari.run()
