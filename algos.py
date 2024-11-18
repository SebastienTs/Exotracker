from napari import Viewer
from napari.types import ImageData, PointsData, LayerDataTuple
from napari.layers import Tracks, Points, Layer
from magicgui.tqdm import trange
from magicgui import magicgui
from skimage.feature import blob_dog as dog
import trackpy as tp
import pandas as pd
import numpy as np
from pathlib import Path
from os import chmod
import pickle
from settings import color_codes
from utils import *
tp.quiet()

#### Spot Detectors

## 1) Multi-scale Difference of Gaussian based (Skimage)

@magicgui(call_button='Detect',
          spot_rad={'widget_type': 'FloatSlider', 'min': 1, 'max': 3},
          detect_thr={'widget_type': 'FloatSlider', 'min': 0.1, 'max': 0.5})
def detect_spots_msdog(vw: Viewer, spot_rad=2, detect_thr=0.3) -> LayerDataTuple:

    if viewer_is_layer(vw, 'Channel1'):

        # Retrieve input image from layer
        img = vw.layers['Channel1'].data

        # Detect blobs frame by frame
        blb_lst, blb_scales, blb_cnt = [], [], 0
        for t in trange(img.shape[0]):

          # Blobs detection
          blobs = dog(img[t, :, :], min_sigma=spot_rad/2, max_sigma=2*np.ceil(spot_rad), sigma_ratio=1.6, threshold=detect_thr*1e-3)
          blb_cnt += len(blobs)

          # Add time frame info and compute blobs local intensity
          coords = np.c_[(t*np.ones(blobs.shape[0]), blobs[:,:-1])]
          stats = disk_int_stats(img,  coords, 3*spot_rad)

          # Keep only valid blobs (right scale + no 0 intensity around) and split coordinates / scales
          blb_coords_scales_kept = [(coords[i, :], blob[-1]) for i, blob in enumerate(blobs)
                      if blob[-1] <= spot_rad and stats['min'][i] >= 1]
          blb_kept_coords, blb_kept_scales = zip(*blb_coords_scales_kept)

          # Accumulate blobs detected in this frame to list
          blb_lst = blb_lst + list(blb_kept_coords)
          blb_scales = blb_scales + list(blb_kept_scales)

        # Display blob count
        detect_spots_msdog.call_button.text = f'Detected Spots ({np.round(len(blb_lst)/img.shape[0])}/frame)'

        return ([blob[:3] for blob in blb_lst],
                {'name': 'Blobs', 'size': int(np.round(5*spot_rad)), 'border_color': color_codes[0],
                 'face_color': 'transparent', 'properties': {'scale': blb_scales}}, 'points')
    else:

        dialogboxmes('Error', 'Load an image first!')

#### Particle Trackers

## 1) TrackPy

@magicgui(call_button='Track',
          search_range={'widget_type': 'IntSlider', 'min': 1, 'max': 5},
          gap_memory={'widget_type': 'IntSlider', 'max': 25},
          min_duration={'widget_type': 'IntSlider', 'min': 5, 'max': 50},
          min_length={'widget_type': 'FloatSlider', 'max': 3},
          max_mean_speed={'widget_type': 'FloatSlider', 'max': 3},
          max_scale={'widget_type': 'FloatSlider', 'min': 1, 'max': 2.5},
          )
def track_spots_pytrack(vw: Viewer, search_range=2, gap_memory=15, min_duration=35, min_length=0, max_mean_speed=0.5, max_scale = 1.33, ) -> LayerDataTuple:

    if viewer_is_layer(vw, 'Blobs'):

        # Retrieve point info from layer
        pts = vw.layers['Blobs'].data
        scales = vw.layers['Blobs'].properties['scale']

        # Convert point info to dataframe and track blobs
        df = pd.DataFrame(np.column_stack((pts, scales)), columns=['frame', 'y', 'x', 'scale'])
        trajectories = tp.link_df(df, search_range=search_range, memory=gap_memory).sort_values(by=['particle', 'frame'])

        # Remove spurious and abnormal blob size / speed tracks
        trajectories_flt = pd.DataFrame(columns=trajectories.columns)
        tracks_total, tracks_kept = 0, 0
        for id, df in trajectories.groupby('particle'):
            duration = df['frame'].iloc[-1]-df['frame'].iloc[0]+1
            length = np.sqrt(df['x'].diff()**2+df['y'].diff()**2).sum()
            scale = df['scale'].mean()
            if (duration >= min_duration and length/duration <= max_mean_speed and length >= min_length and scale <= max_scale):
                trajectories_flt = pd.concat([trajectories_flt, df]).astype(float)
                tracks_kept += 1
            tracks_total += 1

        # Display results summary
        track_spots_pytrack.call_button.text = f'Track Spots ({tracks_kept}/{tracks_total})'

        return (trajectories_flt[['particle', 'frame', 'y', 'x']].values,
                {'name': 'Tracks', 'head_length': 0, 'tail_length': 999, 'tail_width': 3}, 'tracks')

    else:
        dialogboxmes('Error', 'Detect blobs first!')

#### Track Analysis

@magicgui(call_button='Analyze',
          min_start_frame={'widget_type': 'IntSlider', 'max': 25},
          min_preend_frame={'widget_type': 'IntSlider', 'max': 25},
          min_neighbor_dist={'widget_type': 'IntSlider', 'max': 25},
          min_contrast={'widget_type': 'FloatSlider', 'max': 1},
          chan2_contrast_delta={'widget_type': 'FloatSlider', 'max': 1},
          chan2_fraction={'widget_type': 'FloatSlider', 'max': 1})
def analyze_tracks(vw: Viewer, min_start_frame=9, min_preend_frame=25, min_neighbor_dist=4, min_contrast=0.28, chan2_contrast_delta=0.16, chan2_fraction=0.5) -> LayerDataTuple:

  if viewer_is_layer(vw, 'Tracks') and viewer_is_layer(vw, 'Blobs'):

      spot_rad = detect_spots_msdog.spot_rad.value

      # Retrieve image and tracks from layers
      img = vw.layers['Channel1'].data
      trks = vw.layers['Tracks'].data

      # Fill track gaps
      trajectories = pd.DataFrame(trks, columns=['particle', 'frame', 'y', 'x'])
      trajectories = trajectories.groupby('particle', as_index=False).apply(interpolate_track)

      # Flag close sites (including sites re-appearing at close positions at different time points)
      cx = trajectories.groupby('particle', as_index=False)['x'].mean()
      cy = trajectories.groupby('particle', as_index=False)['y'].mean()
      dst_flags = flag_close_points(np.column_stack((cy['y'], cx['x'])), min_neighbor_dist)

      # Filter tracks
      tracks_kept, tracks_kept_props = pd.DataFrame(columns=trajectories.columns), {}
      cnt_tracks, cnt_kept = 0, 0
      for i, df in trajectories.groupby('particle', as_index=False):

          # Temporal information
          lgth = len(df)
          first_frame = df['frame'].iloc[0]
          last_frame = df['frame'].iloc[-1]

          # Chan1 intensity statistics
          diskin = disk_int_stats(img, np.column_stack((df['frame'], df['y'], df['x'])), 1)
          diskout = disk_int_stats(img, np.column_stack((df['frame'], df['y'], df['x'])), int(np.round(1.5*spot_rad)))
          outer = (diskout['mean']*diskout['area']-diskin['mean']*diskin['area'])/(diskout['area']-diskin['area'])
          contrast = (diskin['mean'].mean()/outer.mean())-1
          #part_stats = particle_analysis(img, np.column_stack((df['frame'], df['y'], df['x'])), 4)
          #eccent = part_stats['eccent'].max()

          if ((first_frame >= min_start_frame and last_frame <= (img.shape[0]-min_preend_frame))
                  and dst_flags[cnt_tracks] and contrast >= min_contrast):

              tracks_kept = pd.concat([tracks_kept, df])

              # Store channel 1 intensity profile
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch1_int', list(np.round(diskin['mean'], decimals=1)))
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'track', df)
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'length', lgth)
              cnt_kept += 1

          cnt_tracks += 1
      tracks_kept = tracks_kept.reset_index(drop=True)

      # Channel 2 intensity statistics and foci detection
      cnt_coloc = 0
      if str(load_image_tiff.imagepath2.value).endswith('.tif'):

          # Retrieve channel 2 images
          img2 = vw.layers['Channel2'].data
          #img2_corr = vw.layers['Channel2_corr'].data

          # Classify tracks based on Channel 2 intensity profile
          tracks_chan2_props = dict()
          colors = np.zeros(0, dtype=int)
          for i, df in tracks_kept.groupby('particle', as_index=False):

            # Extend track
            lgth = len(df)
            df = df.reset_index(drop=True)
            if df['frame'][lgth-1]<img2.shape[0]-min_preend_frame:
                df = extend_dataframe_frames(df, min_preend_frame)

            # Extract, analyze and store intensity profiles
            diskin = disk_int_stats(img2, np.column_stack((df['frame'], df['y'], df['x'])), 1)
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch2_int', list(np.round(diskin['mean'], decimals=1)))

            diskout = disk_int_stats(img2, np.column_stack((df['frame'], df['y'], df['x'])), int(np.round(1.5 * spot_rad)))
            outer = (diskout['mean'] * diskout['area'] - diskin['mean'] * diskin['area']) / (diskout['area'] - diskin['area'])
            contrast = np.clip((diskin['mean'] / outer) - 1, 1e-9, 10)
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch2_contrast', list(contrast))

            # Classify channel 2 positive tracks
            delta = contrast.max() - contrast.min()
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch2_positive', delta >= chan2_contrast_delta)

            # Assign channel 2 positive color to blobs
            if delta >= chan2_contrast_delta:
                # Estimate channel 2 track
                start, end, trcklgth = estimate_track_lgth(tracks_kept_props[int(df['particle'].iloc[0])]['ch2_int'], 9, 5, chan2_fraction)
                tracks_chan2_props[int(df['particle'].iloc[0])] = [start, end, trcklgth]
                if lgth-start >= 1 and start >= 1:
                    colors = np.concatenate((colors, np.ones(start, dtype=int)))
                    colors = np.concatenate((colors, 2*np.ones(lgth-start, dtype=int)))
                else:
                    colors = np.concatenate((colors, np.ones(lgth, dtype=int)))
                cnt_coloc += 1
            else:
                colors = np.concatenate((colors, np.zeros(lgth, dtype=int)))

          border_colors = [color_codes[color] for color in colors]

      else:

          border_colors = 'yellow'

      # Display results summary
      analyze_tracks.call_button.text = f'Filter Tracks ({cnt_coloc}/{cnt_kept}/{cnt_tracks})'

      # Hide Blobs and Tracks layers
      vw.layers['Blobs'].visible = False
      vw.layers['Tracks'].visible = False

      # Export results
      with open(Path(load_image_tiff.imagepath.value).with_suffix('.pkl'), 'wb') as file:
          pickle.dump(tracks_kept_props, file)
      if str(load_image_tiff.imagepath2.value).endswith('.tif'):
        with open(Path(load_image_tiff.imagepath2.value).with_suffix('.pkl'), 'wb') as file:
            pickle.dump(tracks_chan2_props, file)
      chmod(str(Path(load_image_tiff.imagepath.value).with_suffix('.pkl')), 0o666)
      chmod(str(Path(load_image_tiff.imagepath2.value).with_suffix('.pkl')), 0o666)

      return ([row[1:] for row in tracks_kept.values], {'name': 'ValidBlobs', 'size': int(np.round(5*spot_rad)),
                                                        'border_color': border_colors, 'face_color': 'transparent'},
                                                        'points')

  else:
      dialogboxmes('Error', 'Track blobs first!')
