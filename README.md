# Exotracker
A napari based application to track exocitosis events in yeast cells and perform related time and dual channel intensity analysis.

The input dataset should be provided as a set of two TIFF time-lapses with these string specifiers:

- **_C1**: exo84 fluorescent label (or equivalent exocyst constitutive protein)
- **_C2**: fluorescent label for a target protein of interest potentially co-localizing with C1 protein

These time-lapses are finely registered, bleaching + background corrected, and possibly slightly noise filtered as an independent pre-processing step.  

Installation
------------

1) Install latest miniconda from https://docs.anaconda.com/miniconda/miniconda-install/ 
2) Clone the repository (or download and unzip it to an empty folder)
3) Copy requirements.txt to your user folder
4) Launch Anaconda Prompt (miniconda3) and type:
	* conda create -n exotracker python=3.9
	* conda activate exotracker
	* pip install -r requirements.txt

Input metadata
--------------

Since the frame step is typically not encoded in TIFF files, it is important to specify it manually  in the "time step" field.
For Trinity (Arrhenius analysis) export, experiment temperature should be specified as _xxC_ in the "proteins" field, for instance: ['uva-30C-exo84', 'uva-30C-sec9'].

Results format
--------------

The results are exported as two picke (.pkl) files in the same folder as the input images (and with same names)

**C1 pickle file**

Each track is an entry (track id) of a dictionnary of dictionnaries with following entries:
'protein1': String combining C1 protein and sample/experimental conditions
'frame_timestep': frame duration
'length': C1 track duration
'track': panda dataframe (columns: particle, frame, y, x) x L rows (time frames)
'ch1_int': intensity profile (vector, length L)

'protein2': String combining C2 protein and sample/experimental conditions
'int_preframe': number of frames before C1 track start (C2 intensity profile analysis)
'int_postframe': number of frames after C1 track end (C2 intensity profile analysis)
'ch1_ext_int': extended intensity profile (vector, length L+int_preframe+int_postframe)
'ch2_ext_int': extended intensity profile (vector, length L+int_preframe+int_postframe)

**C2 pickle file**

C2 tracks are not estimated by a particle tracker as C1 tracks but from a gating algorithm applied to the extended intensity profiles 'ch2_ext_int' extracted at the locations of C1 tracks.

The file holds a Numpy array with 3 columnsas and as many rows as C1 tracks positive for C2 signal:
Columns: C2_track_start_frame, C2_track_end_frame, C2_track_length
Rows: one row per C2 positive C1 track
