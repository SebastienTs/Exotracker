import napari
import warnings
from algos import *
from graphs import *
warnings.filterwarnings("ignore", category=FutureWarning)

# Instantiate Napari and add Widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_images_tiff, area='right', name='Load Images', tabify=False)
dw2 = viewer.window.add_dock_widget(detect_spots_msdog, area='right', name='Detect Spots', tabify=False)
dw3 = viewer.window.add_dock_widget(track_spots_trackpy, area='right', name='Track Spots', tabify=False)
dw4 = viewer.window.add_dock_widget(analyze_tracks_int_gate, area='right', name='Analyze Tracks', tabify=False)
graph_tracks.native.setStyleSheet("background-color: #202020")
graph_tracks.call_button.native.setStyleSheet("""QPushButton {background-color: #002200;color: white;}QPushButton:hover {background-color: #66bb6a;}""")
dw5 = viewer.window.add_dock_widget(graph_tracks, area='right', name='Plot Data', tabify=False)
viewer.window._qt_window.tabifyDockWidget(dw4, dw5)
dw1.setMinimumWidth(360)
dw1.setMaximumWidth(360)

# Display napari viewer
napari.run()
