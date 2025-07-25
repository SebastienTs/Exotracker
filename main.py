import napari
import warnings
from graphs import *
warnings.filterwarnings("ignore", category=FutureWarning)

# Instantiate Napari and add + customize the apperance of Exotracker widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_images_tiff, area='right', name='Load Images', tabify=False)
load_images_tiff.imagepath.native.setStyleSheet("color: pink;")
load_images_tiff.imagepath2.native.setStyleSheet("color: pink;")
load_images_tiff.proteins.native.setStyleSheet("color: violet;")
load_images_tiff.time_step.native.setStyleSheet("color: violet;")
dw2 = viewer.window.add_dock_widget(detect_spots_msdog, area='right', name='Detect Spots', tabify=False)
dw3 = viewer.window.add_dock_widget(track_spots_trackpy, area='right', name='Track Spots', tabify=False)
analyze_tracks_int_gate.call_button.native.setStyleSheet("color: pink;")
analyze_tracks_int_gate.min_startframe.native.setStyleSheet("color: red;")
analyze_tracks_int_gate.min_afterframe.native.setStyleSheet("color: red;")
analyze_tracks_int_gate.min_neighbor_dist.native.setStyleSheet("color: red;")
analyze_tracks_int_gate.min_c1_contrast.native.setStyleSheet("color: red;")
analyze_tracks_int_gate.min_c2_contrast_delta.native.setStyleSheet("color: green;")
analyze_tracks_int_gate.track_c2_int_medrad.native.setStyleSheet("color: green;")
analyze_tracks_int_gate.track_c2_int_thr.native.setStyleSheet("color: green;")
analyze_tracks_int_gate.c2_mode.native.setStyleSheet("color: green;")
analyze_tracks_int_gate.c2_mode.native.setMaximumHeight(36)
dw4 = viewer.window.add_dock_widget(analyze_tracks_int_gate, area='right', name='Analyze Tracks', tabify=False)
curate_tracks.plot_intensity_profiles.native.setStyleSheet("color: pink;")
curate_tracks.model_c2_track.native.setStyleSheet("color: pink;")
curate_tracks.write_ignore_tracks.native.setStyleSheet("color: pink;")
curate_tracks.nplateaus.native.setStyleSheet("padding-left: 40px;")
curate_tracks.steepness.native.setStyleSheet("padding-left: 40px;")
curate_tracks.plot_first_trck.native.setStyleSheet("padding-left: 40px;")
curate_tracks.plot_last_trck.native.setStyleSheet("padding-left: 40px;")
dw5 = viewer.window.add_dock_widget(curate_tracks, area='right', name='Curate Tracks', tabify=False)
tracks_statistics.intnorm.native.setStyleSheet("padding-left: 40px;")
tracks_statistics.std_and_delta.native.setStyleSheet("padding-left: 40px;")
tracks_statistics.time_norm_c1.native.setStyleSheet("padding-left: 40px;")
tracks_statistics.groupfiles.native.setStyleSheet("color: pink;")
tracks_statistics.plot_timelines.native.setStyleSheet("color: pink;")
tracks_statistics.export_to_trinity.native.setStyleSheet("color: pink;")
tracks_statistics.plot_average_intensity_profile.native.setStyleSheet("color: pink;")
tracks_statistics.export_to_trinity.native.setStyleSheet("color: pink;")
dw6 = viewer.window.add_dock_widget(tracks_statistics, area='right', name='Track Statistics', tabify=False)
viewer.window._qt_window.tabifyDockWidget(dw4, dw5)
viewer.window._qt_window.tabifyDockWidget(dw4, dw6)
dw1.setMinimumWidth(360)
dw1.setMaximumWidth(360)

# Display napari viewer
napari.run()
