# Default file names and metadata
filename = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/_TestImages/Test_2x2_pre_C1.tif'
filename2 = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/_TestImages/Test_2x2_pre_C2.tif'
#filename = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/30C/5_Montage/Montage_preprocessed_C1.tif'
#filename2 = 'D:/Projects/UPF/Live_exo84_sec9/Oblique/30C/5_Montage/Montage_preprocessed_C2.tif'
skipfirst, skiplast = 25, 25    # Skip first/last N frame(s)
frame_timestep = 0.123          # Frame length (s)
proteins=['exo84', 'sec9']      # C1/C2 proteins

# Plotting
mx_prefrc = 0.5   # Fractional time before C1 track in average intensity plot
mx_postfrc = 0.5  # Fractional time after C1 track in average intensity plot
color_codes = ['red', 'yellow', 'green'] # Exocytosis events labels: C2-, future C2+, current C2+
