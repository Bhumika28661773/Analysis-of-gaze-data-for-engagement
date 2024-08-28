import h5py
import pandas as pd

# Path to the HDF5 file
file_path = "C:/Users/bhumi/Desktop/eyetracker/gazes/bhumika_3_combine_test_eyetracker_2024-07-26_11h18.31.593.hdf5"

# Open the HDF5 file and extract the BinocularEyeSampleEvent dataset
with h5py.File(file_path, 'r') as hdf:
    binocular_data = hdf['/data_collection/events/eyetracker/BinocularEyeSampleEvent'][:]
    
    # Convert to DataFrame
    binocular_df = pd.DataFrame(binocular_data)

# Save DataFrame to CSV
binocular_csv_path = 'binocular_eye_sample_event1.csv'
binocular_df.to_csv(binocular_csv_path, index=False)

# Display first few rows to verify
print(binocular_df.head())

binocular_csv_path

# Example usage


