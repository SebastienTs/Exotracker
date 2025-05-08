## Default C1/C2 images file paths + imaging related parameters

#filename = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/30C/5_Montage/Montage_preprocessed_C1.tif'
#filename2 = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/30C/5_Montage/Montage_preprocessed_C2.tif'
filename = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/_TestImages/Test_2x2_pre_C1.tif'
filename2 = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/_TestImages/Test_2x2_pre_C2.tif'

start_frame, end_frame = 25, 525            # Analyzed frames (skip start due to instability and end due to bleaching)
c2_preframes, c2_postframes = 0, 25         # Track filter (required for pre- and post- frames intensity based analysis)
frame_timestep = 0.123                      # Frame length (s)
proteins=['exo84', 'sec9']                  # C1/C2 proteins

# Color codes used to label the exocytosis events
color_codes = ['red', 'yellow', 'green']
