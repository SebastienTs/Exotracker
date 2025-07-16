# Default file names and metadata
filename_c1_default = 'D:/Python/data/Exotracker/Mixed_Temps/Montaged_seb_30C_C1_2x2.tif'
filename_c2_default = 'D:/Python/data/Exotracker/Mixed_Temps/Montaged_seb_30C_C2_2x2.tif'
skipfirst_default, skiplast_default = 0, 0              # Don't load first/last N frame(s) from time-lapses
frame_timestep_default = 0.123                          # Frame duration (s)
proteins_default=['uva-30C-exo84', 'uva-30C-sec9']      # C1/C2 porteins experimental conditions

# Plotting
mx_prefrc, mx_postfrc = 1, 1                    # Interval of time (C1 track length normalized) displayed before/after C1 track in average intensity plot
color_codes = ['red', 'yellow', 'green']        # Exocytosis annotation colors: C2-, future C2+, current C2+
