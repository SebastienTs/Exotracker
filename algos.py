from pathlib import Path
from os import chmod, path
from napari import Viewer
from napari.types import ImageData, PointsData, LayerDataTuple
from napari.layers import Tracks, Points, Layer
from magicgui.tqdm import trange
from magicgui import magicgui
from skimage.feature import blob_dog as dog
import trackpy as tp
import pandas as pd
import numpy as np
import pickle
from settings import color_codes
from utils import *
tp.quiet()

#### Image loader

## TIFF images loader widget
@magicgui(call_button='Load',
          imagepath={'widget_type': 'FileEdit', 'label': 'Chan1', 'tooltip': 'Channel 1 image'},
          imagepath2={'widget_type': 'FileEdit', 'label': 'Chan2', 'tooltip': 'Channel 2 image'},
          proteins={'widget_type': 'LineEdit', 'tooltip': 'C1 and C2 proteins names'},
          time_step={'widget_type': 'FloatSpinBox', 'step': 0.001, 'tooltip': 'Frame length (s)'},
          skipfirst={'widget_type': 'IntSlider', 'max': 50, 'tooltip': 'Skip the first N frame(s) when loading the image'},
          skiplast={'widget_type': 'IntSlider', 'max': 50, 'tooltip': 'Skip the last N frame(s) when loading the image'})
def load_images_tiff(vw:Viewer, imagepath=filename, imagepath2=filename2, proteins=proteins, time_step=frame_timestep, skipfirst=skipfirst, skiplast=skiplast):

    viewer_reset(vw)

    if path.isfile(imagepath) and str(imagepath).endswith('.tif'):

        with TiffFile(imagepath) as tif:
            num_pages = len(tif.pages)
            img = imread(imagepath, key=range(min(skipfirst, num_pages-1), max(1, num_pages-skiplast))).astype(np.uint16)

        if path.isfile(imagepath2) and str(imagepath2).endswith('.tif'):
            img2 = imread(imagepath2, key=range(min(skipfirst, num_pages-1), max(1, num_pages-skiplast))).astype(np.uint16)
            if viewer_is_layer(vw, 'Channel2'):
                vw.layers['Channel2'].data = img2
            else:
                vw.add_image(img2, name='Channel2')
            print(f'Loaded image {imagepath2} ({img2.shape})')
        else:
            print("C2 File doesn't exist or isn't a TIFF file")

        vw.add_image(img, name='Channel1')
        print(f'Loaded image {imagepath} ({img.shape})')

    else:

        print("C1 File doesn't exist or isn't a TIFF file")

    return None


#### Blob Detectors

## Multi-scale Difference of Gaussian based (msdog) + parametric filter

@magicgui(call_button='Detect',
          spot_rad={'widget_type': 'FloatSlider', 'min': 1, 'max': 3, 'tooltip': 'Spot detection scale (pixels)'},
          detect_thr={'widget_type': 'FloatSlider', 'min': 0.1, 'max': 1, 'tooltip': 'Spot detection sensitivity threshold'})
def detect_spots_msdog(vw: Viewer, spot_rad=2, detect_thr=0.3) -> LayerDataTuple:

    if viewer_is_layer(vw, 'Channel1'):

        # Fetch input image from napari image layer
        img = vw.layers['Channel1'].data

        # Detect blobs (frame by frame)
        blb_lst, blb_scales = [], []
        for t in trange(img.shape[0]):

          # Blob detection
          blobs = dog(img[t, :, :], min_sigma=spot_rad/2, max_sigma=2*np.ceil(spot_rad), sigma_ratio=1.6, threshold=detect_thr*1e-3)

          # Add time frame info and compute blobs local intensity
          coords = np.c_[(t*np.ones(blobs.shape[0]), blobs[:,:-1])]

          # Keep only valid blobs (<= maximum scale and far from edges) + split spot coordinates / scales
          stats = disk_int_stats(img, coords, 3*spot_rad)
          blb_coords_scales_kept = [(coords[i, :], blob[-1]) for i, blob in enumerate(blobs) if blob[-1] <= spot_rad and stats['min'][i] >= 1]

          if blb_coords_scales_kept:
            blb_kept_coords, blb_kept_scales = zip(*blb_coords_scales_kept)

            # Accumulate blobs from this frame to list
            blb_lst = blb_lst + list(blb_kept_coords)
            blb_scales = blb_scales + list(blb_kept_scales)

        # Display blob count
        print(f'Detected {len(blb_lst)} spots ({np.round(len(blb_lst)/img.shape[0])}/frame)')
        detect_spots_msdog.call_button.text = f'{len(blb_lst)} Spots ({np.round(len(blb_lst)/img.shape[0])}/frame)'

        # Return napari point layer
        return ([blob[:3] for blob in blb_lst],
                {'name': 'Blobs', 'size': int(np.round(5*spot_rad)), 'border_color': 'gray',
                 'face_color': 'transparent', 'properties': {'scale': blb_scales}}, 'points')
    else:

        dialogboxmes('Error', 'Load an image first!')

        return None

#### Particle Trackers

## TrackPy + parametric filter

@magicgui(call_button='Track',
          max_spot_mean_scale={'widget_type': 'FloatSlider', 'min': 1, 'max': 2.5, 'tooltip': 'Track maximum mean spot scale (pixels)'},
          search_range={'widget_type': 'IntSlider', 'min': 1, 'max': 5, 'tooltip': 'Spot search range (pixels)'},
          max_gap={'widget_type': 'IntSlider', 'max': 20, 'tooltip': 'Track maximum gap (frames)'},
          min_duration={'widget_type': 'IntSlider', 'min': 3, 'max': 50, 'tooltip': 'Track minimum duration (frames)'},
          max_duration={'widget_type': 'IntSlider', 'min': 50, 'max': 500, 'tooltip': 'Track maximum duration (frames)'},
          min_length={'widget_type': 'FloatSlider', 'max': 5, 'tooltip': 'Track minimum accumulated length (pixels)'},
          max_mean_speed={'widget_type': 'FloatSlider', 'max': 5, 'tooltip': 'Track maximum average speed (pixels/frame)'})
def track_spots_trackpy(vw: Viewer, max_spot_mean_scale = 1.33, search_range=2, max_gap=15, min_duration=35, max_duration=250, min_length=0, max_mean_speed=0.2) -> LayerDataTuple:

    if viewer_is_layer(vw, 'Blobs'):

        # Fetch blobs from napari point layer
        pts = vw.layers['Blobs'].data
        scales = vw.layers['Blobs'].properties['scale']

        # Convert points to dataframe and track particles
        df = pd.DataFrame(np.column_stack((pts, scales)), columns=['frame', 'y', 'x', 'scale'])
        trajectories = tp.link_df(df, search_range=search_range, memory=max_gap).sort_values(by=['particle', 'frame'])

        # Filter tracks
        trajectories_flt = pd.DataFrame(columns=trajectories.columns)
        tracks_total, tracks_kept = 0, 0
        for id, df in trajectories.groupby('particle'):
            duration = df['frame'].iloc[-1]-df['frame'].iloc[0]+1
            length = np.sqrt(df['x'].diff()**2+df['y'].diff()**2).sum()
            scale = df['scale'].mean()
            if (min_duration <= duration <= max_duration and length/duration <= max_mean_speed
                    and length >= min_length and scale <= max_spot_mean_scale):
                trajectories_flt = pd.concat([trajectories_flt, df]).astype(float)
                tracks_kept += 1
            tracks_total += 1

        # Display results summary
        print(f'Detected {tracks_kept} tracks ({tracks_total} candidates)')
        track_spots_trackpy.call_button.text = f'Track Spots ({tracks_kept}/{tracks_total})'

        # Return napari track layer
        return (trajectories_flt[['particle', 'frame', 'y', 'x']].values,
                {'name': 'Tracks', 'head_length': 0, 'tail_length': 999, 'tail_width': 3}, 'tracks')

    else:

        dialogboxmes('Error', 'Detect blobs first!')

        return None


#### Track Analysis

## Parametric Filter + C2 Intensity Gating

@magicgui(call_button='Filter and Save Tracks',
          min_startframe={'widget_type': 'IntSlider', 'max': 50, 'tooltip': 'Earliest track start (frame index)'},
          min_afterframe={'widget_type': 'IntSlider', 'max': 50, 'tooltip': 'Latest track end (number of frames before last frame)'},
          min_neighbor_dist={'widget_type': 'IntSlider', 'max': 10, 'tooltip': 'Minimum distance to closest point in any other track (pixels)'},
          min_c1_contrast={'widget_type': 'FloatSlider', 'max': 0.5, 'step': 0.001, 'tooltip': 'C1 minimum average contrast'},
          min_c2_contrast_delta={'widget_type': 'FloatSlider', 'max': 0.5, 'step': 0.001, 'tooltip': 'C2 minimum contrast variation'},
          track_c2_int_medrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9, 'tooltip': 'C2 track detection median filter window half length'},
          track_c2_int_thr={'widget_type': 'FloatSlider', 'min': 0.25, 'max': 0.5, 'step': 0.01, 'tooltip': 'C2 track detection normalized intensity threshold'})
def analyze_tracks_int_gate(vw: Viewer, min_startframe=25, min_afterframe=50, min_neighbor_dist=4, min_c1_contrast=0.26,
                            min_c2_contrast_delta=0.15, track_c2_int_medrad=5, track_c2_int_thr=0.35) -> LayerDataTuple:

  if viewer_is_layer(vw, 'Tracks') and viewer_is_layer(vw, 'Blobs'):

      # Fetch spot radius from spot detection widget
      spot_rad = detect_spots_msdog.spot_rad.value

      # Read C1 and C2 protein names from file loader widget
      proteins_str = load_images_tiff.proteins.value
      stripped = proteins_str[1:-1].replace("'", "")
      proteins = [x.strip() for x in stripped.split(',')]

      # Retrieve C1 image and tracks from napari layers
      img = vw.layers['Channel1'].data
      trks = vw.layers['Tracks'].data

      # Fill track gaps
      trajectories = pd.DataFrame(trks, columns=['particle', 'frame', 'y', 'x'])
      trajectories = trajectories.groupby('particle', as_index=False).apply(interpolate_track)

      # Flag closeby tracks
      cx = trajectories.groupby('particle', as_index=False)['x'].mean()
      cy = trajectories.groupby('particle', as_index=False)['y'].mean()
      dst_flags = flag_min_dist(trajectories, min_neighbor_dist)

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
          contrast = np.clip((diskin['mean'].mean()/outer.mean())-1, 1e-9, 1)

          if ((first_frame >= min_startframe and last_frame <= (img.shape[0]-min_afterframe-1))
                  and dst_flags[cnt_tracks] and contrast >= min_c1_contrast):

              tracks_kept = pd.concat([tracks_kept, df])

              # Append track information, track intensity profile and metadata
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'protein1', proteins[0])
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch1_int', list(np.round(diskin['mean'], decimals=1)))
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'track', df)
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'length', lgth)
              tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'frame_timestep', load_images_tiff.time_step.value)

              cnt_kept += 1

          cnt_tracks += 1

      tracks_kept = tracks_kept.reset_index(drop=True)

      # C2 intensity statistics
      cnt_positive, cnt_startcrop, cnt_endcrop = 0, 0, 0
      if str(load_images_tiff.imagepath2.value).endswith('.tif'):

          # Fetch C2 image from napari image layer
          img2 = vw.layers['Channel2'].data
          #img2_corr = vw.layers['Channel2_corr'].data

          # Identify C2 positive C1 tracks
          tracks_chan2_times = dict()
          colors = np.zeros(0, dtype=int)
          for i, df in tracks_kept.groupby('particle', as_index=False):

            # Extend track for C1/C2 pre- and post- intensity analysis (at first/last particle location)
            lgth = len(df)
            df = df.reset_index(drop=True)
            df = extend_dataframe_frames_post(df, min_afterframe)
            df = extend_dataframe_frames_pre(df, min_startframe)

            # Append metadata
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'int_preframe', min_startframe)
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'int_postframe', min_afterframe)

            # Extract and store intensity profile for C1 extended track
            diskin = disk_int_stats(img, np.column_stack((df['frame'], df['y'], df['x'])), 1)
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch1_ext_int', list(np.round(diskin['mean'], decimals=1)))

            # Append metadata
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'protein2', proteins[1])

            # Extract and store intensity and contrast profiles for C2 extended track
            diskin = disk_int_stats(img2, np.column_stack((df['frame'], df['y'], df['x'])), 1)
            diskout = disk_int_stats(img2, np.column_stack((df['frame'], df['y'], df['x'])), int(np.round(1.5 * spot_rad)))
            outer = (diskout['mean'] * diskout['area'] - diskin['mean'] * diskin['area']) / (diskout['area'] - diskin['area'])
            contrast = np.clip((diskin['mean']/outer)-1, 1e-9, 1)
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch2_ext_int', list(np.round(diskin['mean'], decimals=1)))
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch2_ext_contrast', list(contrast))

            # Flag C2 positive tracks (for minimal contrast variation)
            contrast = medfilt(contrast, kernel_size=5)     # Added short temporal filter
            delta = contrast.max()-contrast.min()
            ch2_positive = delta >= min_c2_contrast_delta
            tracks_kept_props = acc_dict(tracks_kept_props, int(df['particle'].iloc[0]), 'ch2_positive', ch2_positive)

            if ch2_positive:

                # Estimate start, end and length of C2 track (from extended intensity profile)
                c2_ext_int = tracks_kept_props[int(df['particle'].iloc[0])]['ch2_ext_int']
                start, end, trcklgth, _, _ = estimate_track_lgth(c2_ext_int, track_c2_int_medrad, track_c2_int_thr)
                tracks_chan2_times[int(df['particle'].iloc[0])] = [start, end, trcklgth]

                # Check if track is cropped (start/end outside analysed interval)
                if start == 0:
                    cnt_startcrop += 1
                if end == (len(c2_ext_int)-1):
                    cnt_endcrop += 1

                # Color-code C1 blobs according to C2 positiveness
                lgth_negative = min(max(start-min_startframe, 0), lgth)
                colors = np.concatenate((colors, np.ones(lgth_negative, dtype=int)))
                lgth_positive = max(min(trcklgth, lgth-lgth_negative), 0)
                colors = np.concatenate((colors, 2*np.ones(lgth_positive, dtype=int)))
                lgth_remaining = max(lgth-lgth_negative-lgth_positive, 0)
                colors = np.concatenate((colors, np.ones(lgth_remaining, dtype=int)))

                cnt_positive += 1
            else:
                colors = np.concatenate((colors, np.zeros(lgth, dtype=int)))

          border_colors = [color_codes[color] for color in colors]

      else:

          border_colors = color_codes[0]

      # Display results summary
      print(f'Filtered C1 tracks (Total)      : {cnt_kept}')
      print(f'Filtered C1 tracks (C2 positive): {cnt_positive} (cropped start/stop: {cnt_startcrop}/{cnt_endcrop})')
      analyze_tracks_int_gate.call_button.text = f'Filter Tracks ({cnt_positive}/{cnt_kept}/{cnt_tracks})'

      # Hide Blobs and Tracks layers
      vw.layers['Blobs'].visible = False
      vw.layers['Tracks'].visible = False

      # Export results
      tracks_props_file = Path(load_images_tiff.imagepath.value).with_suffix('.pkl')
      tracks_chan2_times_file = Path(load_images_tiff.imagepath2.value).with_suffix('.pkl')
      with open(tracks_props_file, 'wb') as file:
          pickle.dump(tracks_kept_props, file)
          chmod(str(tracks_props_file), 0o666)
      if str(load_images_tiff.imagepath2.value).endswith('.tif'):
        with open(tracks_chan2_times_file, 'wb') as file:
            pickle.dump(tracks_chan2_times, file)
            chmod(str(tracks_chan2_times_file), 0o666)

      # Return color-coded blobs as napari point layer
      return ([row[1:] for row in tracks_kept.values],
              {'name': 'ValidBlobs', 'size': int(np.round(5*spot_rad)), 'border_color': border_colors, 'face_color': 'transparent'}, 'points')

  else:

      dialogboxmes('Error', 'Track blobs first!')

      return None
