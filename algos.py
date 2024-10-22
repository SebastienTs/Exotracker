from napari.types import ImageData, PointsData, LayerDataTuple
from napari import Viewer
from magicgui import magicgui
from magicgui.tqdm import trange
from napari.layers import Tracks, Points, Layer
from skimage.feature import blob_dog as dog
import numpy as np
import pandas as pd
import trackpy as tp
from settings import color_codes
from utils import *

#### Spot Detectors

## 1) Multi-scale Difference of Gaussian based (Skimage)

@magicgui(call_button='Detect Spots',
          spot_rad={'widget_type': 'FloatSlider', 'max': 5},
          min_intensity={'widget_type': 'FloatSlider', 'min': 0.1, 'max': 0.5})
def detect_spots_msdog(vw: Viewer, spot_rad=2, min_intensity=0.3) -> LayerDataTuple:

    # Input image
    if viewer_is_layer(vw, 'Channel1'):
        img = vw.layers['Channel1'].data

        # Detect blobs frame by frame
        blb_lst, blb_props, blb_stats, blb_cnt = [], [], [], 0
        for t in trange(img.shape[0]):
          blobs = dog(img[t, :, :], min_sigma=spot_rad/2, max_sigma=2*np.ceil(spot_rad), sigma_ratio=1.6, threshold=min_intensity*1e-3)
          blobs_coords = np.c_[(t*np.ones(blobs.shape[0]), blobs[:,:-1])]
          stats = disk_int_stats(img,  blobs_coords, 3*spot_rad, False)
          blb_kept =  [np.insert(blob[:2], 0, t).astype(int) for i, blob in enumerate(blobs)
                       if blob[2] <= spot_rad and stats['min'][i] >= 1]
          blb_newprops = [blob[2] for i, blob in enumerate(blobs)
                          if blob[2] <= spot_rad and stats['min'][i] >= 1]
          blb_lst = blb_lst + blb_kept
          blb_props = blb_props + blb_newprops
          blb_cnt += len(blobs)

        detect_spots_msdog.call_button.text = f'Detected Spots ({np.round(len(blb_lst)/img.shape[0])}/frame)'

        return ([blob[:3] for blob in blb_lst],
                {'name': 'Blobs', 'size': int(np.round(4*spot_rad+1)), 'border_color': color_codes[0], 'face_color': 'transparent',
                 'properties': {'scale': blb_props}}, 'points')
    else:
        dialogboxmes('Error', 'No Channel1 layer!')

#### Particle Trackers

## 1) TrackPy

@magicgui(call_button='Track',
          search_range={'widget_type': 'IntSlider', 'max': 5},
          max_mean_speed={'widget_type': 'FloatSlider', 'max': 2},
          max_scale={'widget_type': 'FloatSlider', 'max': 2},
          max_gaps={'widget_type': 'IntSlider', 'max': 25},
          min_duration={'widget_type': 'IntSlider', 'min': 15, 'max': 60})
def track_spots_pytrack(vw: Viewer, search_range=2, max_mean_speed=0.5, max_scale = 1.33,
                        max_gaps=15, min_duration=35) -> LayerDataTuple:

    if viewer_is_layer(vw, 'Blobs'):

        # Input data points
        pts = vw.layers['Blobs'].data
        scales = vw.layers['Blobs'].properties['scale']

        # Convert to dataframe and track particles
        df = pd.DataFrame(np.column_stack((pts, scales)), columns=['frame', 'y', 'x', 'scale'])
        tp.quiet()
        trajectories = tp.link_df(df, search_range=search_range, memory=max_gaps).sort_values(by=['particle', 'frame'])

        # Filter tracks
        trajectories_flt = pd.DataFrame(columns=trajectories.columns)
        tracks_total, tracks_kept = 0, 0
        for id, df in trajectories.groupby('particle'):
            duration = df['frame'].iloc[-1]-df['frame'].iloc[0]+1
            length = np.sqrt(df['x'].diff()**2+df['y'].diff()**2).sum()
            scale = df['scale'].mean()
            if (duration >= min_duration) and (length/duration <= max_mean_speed) and scale <= max_scale:
                trajectories_flt = pd.concat([trajectories_flt, df]).astype(float)
                tracks_kept += 1
            tracks_total += 1
        track_spots_pytrack.call_button.text = f'Track Spots ({tracks_kept}/{tracks_total})'

        return (trajectories_flt[['particle', 'frame', 'y', 'x']].values,
                {'name': 'Tracks', 'head_length': 0, 'tail_length': 999, 'tail_width': 3}, 'tracks')

    else:
        dialogboxmes('Error', 'No Blobs layer!')

#### Properties + context-based track filter

@magicgui(call_button='Filter Tracks',
          min_start_frame={'widget_type': 'IntSlider', 'max': 25},
          max_end_frame={'widget_type': 'IntSlider', 'max': 25},
          min_neighbor_dist={'widget_type': 'IntSlider', 'max': 25},
          min_contrast={'widget_type': 'FloatSlider', 'max': 1},
          chan2_ext={'widget_type': 'IntSlider', 'max': 15},
          chan2_delta={'widget_type': 'FloatSlider', 'max': 1})
def filter_tracks(vw: Viewer, min_start_frame=15, max_end_frame=15, min_neighbor_dist=4, min_contrast=0.29,
                  chan2_ext=5, chan2_delta=0.1) -> LayerDataTuple:

  if viewer_is_layer(vw, 'Tracks') and viewer_is_layer(vw, 'Blobs'):

      spot_rad = detect_spots_msdog.spot_rad.value

      # Fill gaps in tracks
      img = vw.layers['Channel1'].data
      trks = vw.layers['Tracks'].data
      trajectories = pd.DataFrame(trks, columns=['particle', 'frame', 'y', 'x'])
      trajectories = trajectories.groupby('particle', as_index=False).apply(interpolate_track)

      # Flag close track centroids
      cx = trajectories.groupby('particle', as_index=False)['x'].mean()
      cy = trajectories.groupby('particle', as_index=False)['y'].mean()
      ct = trajectories.groupby('particle', as_index=False)['frame'].mean()
      coords = np.column_stack((np.zeros(len(cy['y'])), cy['y'], cx['x']))
      dst_flags = flag_close_points(coords, min_neighbor_dist)

      # Filter tracks
      tracks_kept = pd.DataFrame(columns=trajectories.columns)
      tracks_discarded = pd.DataFrame(columns=trajectories.columns)
      cnt_tracks, cnt_kept = 0, 0
      for i, df in trajectories.groupby('particle', as_index=False):
          first_frame = df['frame'].iloc[0]
          last_frame = df['frame'].iloc[-1]

          # Chan1 intensity statistics
          diskin = disk_int_stats(img, np.column_stack((df['frame'], df['y'], df['x'])), int(np.round(0.5*spot_rad)), False)
          diskout = disk_int_stats(img, np.column_stack((df['frame'], df['y'], df['x'])), int(np.round(1.5*spot_rad)), False)
          outer = (diskout['mean']*diskout['area']-diskin['mean']*diskin['area'])/(diskout['area']-diskin['area'])
          contrast = (diskin['mean'].mean()/outer.mean())-1
          #eccent = diskout['eccent'].max()
          if ((first_frame >= min_start_frame and last_frame <= (img.shape[0]-max_end_frame)) and
                  dst_flags[cnt_tracks] and contrast >= min_contrast):
              tracks_kept = pd.concat([tracks_kept, df])
              cnt_kept += 1
          else:
              tracks_discarded = pd.concat([tracks_discarded, df])
          cnt_tracks += 1

      # Chan2 intensity statistics
      cnt_coloc = 0
      if load_image_tiff.loadchan2.value:
          img2 = vw.layers['Channel2'].data
          # pre-subtract first frame
          img2 = (img2 + img2[0, :, :].max() + 1) - img2[0, :, :]
          colors = np.zeros(0, dtype=int)
          for i, df in tracks_kept.groupby('particle', as_index=False):
            df = df.reset_index(drop=True)
            lgth = len(df)
            if df['frame'][len(df)-1]<img2.shape[0]-chan2_ext:
                df = extend_dataframe_frames(df, chan2_ext)
            diskin = disk_int_stats(img2, np.column_stack((df['frame'], df['y'], df['x'])), int(spot_rad), False)
            delta = (diskin['mean'].max() / (diskin['mean'].min() + 1e-9)) - 1
            cnt_coloc += int(delta >= chan2_delta)
            colors = np.concatenate((colors, (delta >= chan2_delta)*np.ones(lgth)))
          border_colors = np.where(colors, color_codes[2], color_codes[1]).tolist()
      else:
          border_colors = 'yellow'

      filter_tracks.call_button.text = f'Filter Tracks ({cnt_coloc}/{cnt_kept}/{cnt_tracks})'

      # Hide Blobs and Tracks layers
      vw.layers['Blobs'].visible = False
      vw.layers['Tracks'].visible = False

      return ([row[1:] for row in tracks_kept.values], {'name': 'ValidBlobs', 'size': 9, 'border_color': border_colors,
                    'face_color': 'transparent'}, 'points')

  else:
      dialogboxmes('Error', 'No Tracks + Blobs layers!')
