# Exotracker
A python script to track exocitosis events in yeast cells and perform all kind of related measurements
The expected input dataset is a dual color two time-lapse with:
- C1: exo84 label (or equivalent constitutive protein)
- C2: protein label of interest (expected to partially temporally co-localize with C1 protein)

Installation
------------

1) Install latest miniconda: https://docs.anaconda.com/miniconda/miniconda-install/ 
2) Clone the repository (or download and unzip to an empty folder)
3) Copy requirements.txt to your user folder
4) Launch Anaconda Prompt (miniconda3) and type:
	* conda create -n exotracker python=3.9
	* conda activate exotracker
	* pip install -r requirements.txt

Input format
------------

The input images should be two TIFF files with _C1 and _C2 specifiers, one for each fluorescence channels
For Trinity export, temperatures should be specified as a _xxC_ field in the protein condition strings

Results format
--------------

The results are exported as two .pkl files (same folders and names as the input images)

C1 pkl file

Each track is an entry (track id) in a dictionnary of dictionnaries with the following entries:
'protein1': String combining C1 protein name and experimental conditions
'length': C1 track duration
'ch1_int': intensity profile (vector, length L)
'track': panda dataframe (columns: particle, frame, y, x) x L rows
'frame_timestep': frame duration

'protein2': String combining C2 protein name and experimental conditions
'int_preframe': number of frames before C1 track start for extended intensity profile analysis
'int_postframe': number of frames after C1 track end for extended intensity profile analysis
'ch1_ext_int': intensity array (length L+int_preframe+int_postframe)
'ch2_ext_int': intensity array (length L+int_preframe+int_postframe)

C2 pkl file

The file holds a Numpy array of shape Nx3 with:
Columns: C2_track_start_frame, C2_track_end_frame, C2_track_length
Rows: one row per C2 positive C1 track

C2 tracks start/end/length are estimated from the EXTENDED intensity profiles 'ch2_ext_int'
