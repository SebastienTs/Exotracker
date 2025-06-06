import napari
import warnings
from algos import *
from graphs import *
warnings.filterwarnings("ignore", category=FutureWarning)

# Instantiate Napari and add Widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_images_tiff, area='right', name='Load Images', tabify=False)
load_images_tiff.proteins.native.setStyleSheet("color: violet;")
load_images_tiff.time_step.native.setStyleSheet("color: violet;")
dw2 = viewer.window.add_dock_widget(detect_spots_msdog, area='right', name='Detect Spots', tabify=False)
dw3 = viewer.window.add_dock_widget(track_spots_trackpy, area='right', name='Track Spots', tabify=False)
dw4 = viewer.window.add_dock_widget(analyze_tracks_int_gate, area='right', name='Filter & Save Tracks', tabify=False)
dw5 = viewer.window.add_dock_widget(curate_tracks, area='right', name='Curate Track', tabify=False)
tracks_statistics.intnorm.native.setStyleSheet("padding-left: 40px;")
tracks_statistics.plot_average_intensity_profile.native.setStyleSheet("color: violet;")
tracks_statistics.native.setStyleSheet("background-color: #202020")
tracks_statistics.export_to_trinity.native.setStyleSheet("color: violet;")
tracks_statistics.call_button.native.setStyleSheet("""QPushButton {background-color: #002200;color: white;}QPushButton:hover {background-color: #66bb6a;}""")
dw6 = viewer.window.add_dock_widget(tracks_statistics, area='right', name='Track Statistics', tabify=False)
viewer.window._qt_window.tabifyDockWidget(dw4, dw5)
viewer.window._qt_window.tabifyDockWidget(dw4, dw6)
dw1.setMinimumWidth(360)
dw1.setMaximumWidth(360)

# Display napari viewer
napari.run()
