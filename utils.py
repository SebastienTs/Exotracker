#from spotiflow.model import Spotiflow
from magicgui import magicgui
from napari import Viewer
from napari.types import LayerDataTuple, PointsData, TracksData
from settings import filename, filename2, start_frame, end_frame
from tifffile import imread
from skimage.filters import gaussian
from scipy import ndimage
import pandas as pd
import numpy as np
import ctypes
import math
import matplotlib.pyplot as plt
from scipy.signal import medfilt

#### Utility functions

# Return z,y,x coordinates of the inner points of an origin centered disk
def disk_pts(rad):

  pts, ptsq = [], [[], [], [], []]
  rad = np.round(rad).astype(int)
  for x in range(-rad,rad+1,1):
    for y in range(-rad,rad+1,1):
      if (x**2+y**2) <= rad**2:
        pts.append([0, y, x])
        ptsq[2*(x>0)+(y>0)].append([0, y, x])

  return pts, ptsq


# Compute min/mean/max image intensity within disks centered on input points
def disk_int_stats(img, pts, rad):

  rad = np.round(rad).astype(int)
  pts = np.round(pts).astype(int)
  offsets, offsetsq = disk_pts(rad)
  min_int,  mean_int, max_int = (np.full(len(pts), np.NaN), np.full(len(pts), np.NaN),
                                 np.full(len(pts), np.NaN))
  for i, pt in enumerate(pts):
    if (pt[1] >= rad) and (pt[2] >= rad) and (pt[1] < img.shape[1]-rad) and (pt[2] < img.shape[2]-rad):
        coords = np.round(np.array(pt)+np.array(offsets)).astype(int)
        min_int[i] = np.min(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        mean_int[i] = np.mean(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        max_int[i] = np.max(img[coords[:, 0], coords[:, 1], coords[:, 2]])

  return {'min': min_int, 'mean': mean_int, 'max': max_int, 'area': len(offsets)}


# Compute the intensity moments of an image
def intensity_moments(img):

    y, x = np.indices(img.shape)
    m00 = np.sum(img)
    m10, m01 = np.sum(x*img), np.sum(y*img)
    m20, m02, m11 = np.sum(x**2*img), np.sum(y**2*img), np.sum(x*y*img)
    x_c, y_c = m10/m00, m01/m00
    mu20, mu02,mu11 = m20/m00-x_c**2, m02/m00-y_c**2, m11/m00-x_c*y_c

    return mu20, mu02, mu11


# Report intensity statistics of particles centered on input points (segmentationless)
def particle_analysis(img, pts, rad):

    rad = np.round(rad).astype(int)
    pts = np.round(pts).astype(int)
    density, eccent, ratio, theta = (np.full(len(pts), np.NaN), np.full(len(pts), np.NaN),
                                     np.full(len(pts), np.NaN) , np.full(len(pts), np.NaN))
    if rad >= 4:
        for i, pt in enumerate(pts):
            crop = img[pt[0], pt[1]-rad:pt[1]+rad, pt[2]-rad:pt[2]+rad].astype(float)
            mu20, mu02, mu11 = intensity_moments(crop)
            major = np.sqrt(2*(mu20+mu02+np.sqrt(4*mu11**2+(mu20-mu02)**2)))
            minor = np.sqrt(2*(mu20+mu02-np.sqrt(4*mu11**2+(mu20-mu02)**2)))
            ratio[i] = minor/major
            eccent[i] = np.sqrt(1-(minor/major)**2)
            density[i] = np.clip(1-(crop[:1,:].mean()+crop[-1:,:].mean()+crop[:,:1].mean()+crop[:,-1:].mean()+1e-9)
                              /(4*crop[2:-2, 2:-2].mean()), 0, 1)
            if ratio[i] > 0.95:
              theta[i] = np.nan
            else:
              theta[i] = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    return {'density': np.round(density, decimals=2), 'eccent': np.round(eccent, decimals=2),
            'ratio': np.round(ratio, decimals=2), 'theta': np.round(theta, decimals=2)}


# Fill gaps from a dataframe track
def interpolate_track(group):

    frames = pd.RangeIndex(group['frame'].min(), group['frame'].max() + 1)
    group_interp = group.set_index('frame', drop=False).reindex(frames)
    group_interp['particle'] = group_interp['particle'].ffill()
    group_interp[['x', 'y', 'frame']] = group_interp[['x', 'y', 'frame']].interpolate(method='linear')

    return group_interp


# Add extra frames to a dataframe track (same position as last frame)
def extend_dataframe_frames(df, nrows):

    for j in range(nrows):
        df = pd.concat([df, pd.DataFrame(df.iloc[-1]).T], ignore_index=True)
        df.loc[df.index[-1], 'frame'] += 1
    return df


# Flag points which closest nieghbor is below a distance threshold
def flag_close_points(pts, min_dst):

  dsts = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=-1))
  np.fill_diagonal(dsts, np.inf)
  dst_flags = np.all(dsts > min_dst, axis=1)

  return dst_flags


# Add a subkey to an entry of a dictionary
def acc_dict(dct, key, subkey, value):

    if key not in dct:
        dct[key] = {}
    dct[key][subkey] = value
    return dct


# Find the position and length of the longest non zero sequence in a 1D array
def longest_non_zero_sequence(seq):
    st, en = max(((i, i + len(s)) for i, s in enumerate(''.join(map(str, seq)).split('0')) if s),
                     key=lambda x: x[1] - x[0], default=(0, 0))
    return st, en, en - st


# Find longest high level plateau in an intensity profile
def estimate_track_lgth(int_profile, medrad1, medrad2, thr):

    vals = np.array(int_profile)
    vals_flt = medfilt(vals, kernel_size=medrad1)
    vals_flt = (vals_flt - vals_flt.min()) / (vals_flt.max() - vals_flt.min())
    trackflag = (vals_flt >= thr).astype(int)
    trackflag = medfilt(trackflag, kernel_size=medrad2)
    start, end , lgth = longest_non_zero_sequence(trackflag)

    return int(start), int(end), int(lgth)

#### Napari widgets / dialog boxes

# Image loader widget
@magicgui(call_button='Load',
          imagepath={'widget_type': 'FileEdit', 'label': 'Chan1'},
          imagepath2={'widget_type': 'FileEdit', 'label': 'Chan2'},
          start_frame={'widget_type': 'IntSlider', 'max': 999},
          end_frame={'widget_type': 'IntSlider', 'max': 999})
def load_image_tiff(vw:Viewer, imagepath=filename, imagepath2=filename2, start_frame=start_frame, end_frame=end_frame):

    img = imread(imagepath, key=range(start_frame, end_frame)).astype(np.uint16)

    if str(imagepath2).endswith('.tif'):
        img2 = imread(imagepath2, key=range(start_frame, end_frame)).astype(np.uint16)
        img2_proc = ((img2 + img2[0, :, :].max() + 1)
                     - gaussian(img2[0, :, :], sigma=5, preserve_range=True)).astype(np.uint16)
        if viewer_is_layer(vw, 'Channel2'):
            vw.layers['Channel2'].data = img2
            vw.layers['Channel2_proc'].data = img2_proc
        else:
            vw.add_image(img2, name='Channel2')
            vw.add_image(img2_proc, name='Channel2_proc')
        print(f'Loaded image {imagepath2} ({img2.shape})')

    if viewer_is_layer(vw, 'Channel1'):
        vw.layers['Channel1'].data = img
    else:
        vw.add_image(img, name='Channel1')

    print(f'Loaded image {imagepath} ({img.shape})')

    return None


# Display message dialog box
def dialogboxmes(message, title):
    return ctypes.windll.user32.MessageBoxW(0, title, message, 0)

#### Napari layers utilities


# Check if viewer layer with specific name exists
def viewer_is_layer(vw: Viewer, layername):

    found = False
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername: found = True

    return found


# Close all viewer layers holding the string layername in their name
def viewer_close_layer(vw: Viewer, layername):

    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername:
                vw.layers.pop(i)


## Spotiflow

# Spotiflow widget
@magicgui(call_button='Detect')
def detect_spots_spotiflow(vw: Viewer) -> LayerDataTuple:

    if viewer_is_layer(vw, 'Channel1'):

        # Input image
        img = vw.layers['Channel1'].data

        # Model
        model = Spotiflow.from_pretrained("general")

        # Detect blobs frame by frame
        blb_lst = []
        for t in range(img.shape[0]):
            points, details = model.predict(img[t, :, :])
            blb_lst = blb_lst + [np.insert(blob, 0, t).astype(int) for blob in points]

        return (blb_lst,{'name': 'Blobs', 'size': 9, 'border_color': 'red', 'face_color': 'transparent'}, 'points')

    else:
        dialogboxmes('Error', 'No Channel1 layer!')
