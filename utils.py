#from spotiflow.model import Spotiflow
from magicgui import magicgui
from napari import Viewer
from napari.types import LayerDataTuple, PointsData, TracksData
from tifffile import imread
from scipy import ndimage
import pandas as pd
import numpy as np
import ctypes
from settings import filename, filename2, start_frame, end_frame, loadchan2

#### Utility functions

def disk_pts(rad):
  pts, ptsq = [], [[], [], [], []]
  rad = np.round(rad).astype(int)
  for x in range(-rad,rad+1,1):
    for y in range(-rad,rad+1,1):
      if (x**2+y**2) <= rad**2:
        pts.append([0, y, x])
        ptsq[2*(x>0)+(y>0)].append([0, y, x])
  return pts, ptsq


def intensity_moments(image):
    y, x = np.indices(image.shape)
    m00 = np.sum(image)
    m10, m01 = np.sum(x*image), np.sum(y*image)
    m20, m02, m11 = np.sum(x**2*image), np.sum(y**2*image), np.sum(x*y*image)
    x_c, y_c = m10/m00, m01/m00
    mu20, mu02,mu11 = m20/m00-x_c**2, m02/m00-y_c**2, m11/m00-x_c*y_c
    return mu20, mu02, mu11


def particle_eccent(image):
    mu20, mu02, mu11 = intensity_moments(image)
    #theta = 0.5 * np.arctan2(2*mu11, mu20-mu02)
    major = np.sqrt(2*(mu20+mu02+np.sqrt(4*mu11**2+(mu20-mu02)**2)))
    minor = np.sqrt(2*(mu20+mu02-np.sqrt(4*mu11**2+(mu20-mu02)**2)))
    eccent = np.sqrt(1-(minor/major)**2)
    return {'eccent': eccent}


def disk_int_stats(img, pts, rad, charpart):
  rad = np.round(rad).astype(int)
  pts = np.round(pts).astype(int)
  offsets, offsetsq = disk_pts(rad)
  min_int,  mean_int, max_int = np.full(len(pts), np.NaN), np.full(len(pts), np.NaN), np.full(len(pts), np.NaN)
  eccent = np.full(len(pts), np.NaN)
  for i, pt in enumerate(pts):
    if (pt[1] >= rad) and (pt[2] >= rad) and (pt[1] < img.shape[1]-rad) and (pt[2] < img.shape[2]-rad):
        coords = np.round(np.array(pt)+np.array(offsets)).astype(int)
        min_int[i] = np.min(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        mean_int[i] = np.mean(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        max_int[i] = np.max(img[coords[:, 0], coords[:, 1], coords[:, 2]])
        if charpart:
            img_crop = img[pt[0], pt[1]-rad:pt[1]+rad, pt[2]-rad:pt[2]+rad].astype(float)
            part_stats = particle_eccent(img_crop)
            eccent[i] = part_stats['eccent']
  return {'min': min_int, 'mean': mean_int, 'max': max_int, 'eccent': eccent, 'area': len(offsets)}


def interpolate_track(group):
    frames = pd.RangeIndex(group['frame'].min(), group['frame'].max() + 1)
    group_interp = group.set_index('frame', drop=False).reindex(frames)
    group_interp['particle'] = group_interp['particle'].ffill()
    group_interp[['x', 'y', 'frame']] = group_interp[['x', 'y', 'frame']].interpolate(method='linear')
    return group_interp


def flag_close_points(pts, min_dst):
  dsts = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=-1))
  np.fill_diagonal(dsts, np.inf)
  dst_flags = np.all(dsts > min_dst, axis=1)
  return dst_flags


def extend_dataframe_frames(df, nrows):
    for j in range(nrows):
        df = pd.concat([df, pd.DataFrame(df.iloc[-1]).T], ignore_index=True)
        df.loc[df.index[-1], 'frame'] += 1
    return df

#### Napari widgets / dialog boxes


@magicgui(call_button='Load Images',
          imagepath={'widget_type': 'FileEdit', 'label': 'Chan1'},
          imagepath2={'widget_type': 'FileEdit', 'label': 'Chan2'},
          start_frame={'widget_type': 'IntSlider', 'max': 999},
          end_frame={'widget_type': 'IntSlider', 'max': 999},
          loadchan2={'widget_type': 'CheckBox'})
def load_image_tiff(vw:Viewer, imagepath=filename, imagepath2=filename2, start_frame=start_frame, end_frame=end_frame,
                    loadchan2=loadchan2):

    img = imread(imagepath, key=range(start_frame, end_frame))

    if loadchan2:
        img2 = imread(imagepath2, key=range(start_frame, end_frame))
        if viewer_is_layer(vw, 'Channel2'):
            vw.layers['Channel2'].data = img2
        else:
            vw.add_image(img2, name='Channel2')
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

# Return True if a viewer layer with layrname is found
def viewer_is_layer(vw: Viewer, layername):
    found = False
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername: found = True
    return found

# Close all viewer layers with layername in name
def viewer_close_layer(vw: Viewer, layername):
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername:
                vw.layers.pop(i)


## Spotiflow

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
