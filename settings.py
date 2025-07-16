# Default file names and metadata
filename_c1_default = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/Su_Mixed_Temps_Sec9/Montaged_seb_30C_C1_2x2.tif'
filename_c2_default = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/Su_Mixed_Temps_Sec9/Montaged_seb_30C_C2_2x2.tif'
skipfirst_default, skiplast_default = 0, 0              # Don't load first/last N frame(s) from time-lapses
frame_timestep_default = 0.123                          # Frame duration (s)
proteins_default=['uva-30C-exo84', 'uva-30C-sec9']      # C1/C2 experimental conditions

# Plotting
mx_prefrc, mx_postfrc = 1, 1                    # C1 norm. time displayed before/after C1 track (average intensity plot)
color_codes = ['red', 'yellow', 'green']        # Exocytosis annotation colors: C2-, future C2+, current C2+

#### Nomenclature
#
# Contrast: Intensity ratio between spot intensity (1 pixel radius disk) and surrounding (1.5*spot_rad radius disk)
#
#### Input format
#
# Input image(s) must be TIFF files with the _C1 and _C2 specifier for the fluorescence channels
# For Trinity export it is also required that the temperature be specified as a _xxC_ field in the protein conditions
#
#### Results format
#
# The results are exported to two .pkl files (same folder(s) as input images)
#
## C1 pkl file
#
# Each track is an entry (track id) in a dictionnary of dictionnaries with the following entries:
# 'protein1': String combining C1 protein name and experimental conditions
# 'length': C1 track length
# 'ch1_int': intensity profile (vector, length L)
# 'track': panda dataframe (columns: particle, frame, y, x) x L rows
# 'frame_timestep': frame duration
#
# 'protein2': String combining C2 protein name and experimental conditions
# 'int_preframe': number of frames before C1 track start for extended intensity profile analysis
# 'int_postframe': number of frames after C1 track end for extended intensity profile analysis
# 'ch1_ext_int': intensity array (length L+int_preframe+int_postframe)
# 'ch2_ext_int': intensity array (length L+int_preframe+int_postframe)
#
## C2 pkl file
#
# The file contains a Numpy array of shape Nx3
# Columns: C2_track_start_frame, C2_track_end_frame, C2_track_length
# Rows: one row per C2 positive C1 track
# C2 tracks start/end/length are estimated from the EXTENDED intensity profiles 'ch2_ext_int'!
#
####
