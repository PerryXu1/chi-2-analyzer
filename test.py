import os
import glob

# Non-recursive search: only look in the current directory
pattern = "AC_frequency_dependence_ps_*_max*"
files = glob.glob(pattern)

print(f"Found {len(files)} files in the current directory to rename.")

for old_filepath in files:
    filename = os.path.basename(old_filepath)
    
    # Remove "_ps_" from the filename
    new_filename = filename.replace("_ps_", "_")
    new_filepath = new_filename
    
    try:
        if old_filepath != new_filepath:
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
        else:
            print(f"Skipped (no change needed): {filename}")
            
    except Exception as e:
        print(f"Error renaming {old_filepath}: {e}")

print("Batch rename complete!")