import cv2
import pandas as pd
import numpy as np
import os

# Read the video
input_video_path = 'video3.mp4'  # Replace with your input video file path
output_video_path = 'output_video_with_gaze.mp4'  # Path to save the output video


# Read gaze data from Excel file
excel_file_path = 'gazes/Bhumika_3_C.xlsx'  # Replace with your Excel file path

# Check if the input video file exists
if not os.path.exists(input_video_path):
    print(f"Video file not found: {input_video_path}")
    exit()

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error opening video file: {input_video_path}")
    exit()

# Verify video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video resolution
video_width = frame_width
video_height = frame_height

print(f"Video properties - Width: {frame_width}, Height: {frame_height}, Frame count: {frame_count}, FPS: {fps}")


# Check if the file exists
if not os.path.exists(excel_file_path):
    print(f"File not found: {excel_file_path}")
    exit()

try:
    df = pd.read_excel(excel_file_path, engine='openpyxl')
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# Ensure the required columns are present
required_columns = ['time', 'left_gaze_x', 'left_gaze_y', 'right_gaze_x', 'right_gaze_y']
if not all(column in df.columns for column in required_columns):
    print(f"Missing required columns in the Excel file. Required columns are: {required_columns}")
    exit()

# Convert 'time' to numeric and adjust if necessary (e.g., to milliseconds)
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['time'] = df['time'] * 1000  # Convert to milliseconds if necessary

# Calculate the average gaze points
df['avg_gaze_x'] = (df['left_gaze_x'] + df['right_gaze_x']) / 2
df['avg_gaze_y'] = (df['left_gaze_y'] + df['right_gaze_y']) / 2

# Determine the minimum and maximum values
x_min, y_min = df[['avg_gaze_x', 'avg_gaze_y']].min()
x_max, y_max = df[['avg_gaze_x', 'avg_gaze_y']].max()

# Normalize the average gaze points
df['norm_gaze_x'] = (df['avg_gaze_x'] - x_min) / (x_max - x_min)
df['norm_gaze_y'] = (df['avg_gaze_y'] - y_min) / (y_max - y_min)

# Map normalized coordinates to video coordinates
df['video_gaze_x'] = df['norm_gaze_x'] * video_width
df['video_gaze_y'] = df['norm_gaze_y'] * video_height

# Apply offset to center the gaze points (if necessary)
center_x = video_width / 2
center_y = video_height / 2

current_center_x = df['video_gaze_x'].mean()
current_center_y = df['video_gaze_y'].mean()

offset_x = center_x - current_center_x
offset_y = center_y - current_center_y

df['centered_gaze_x'] = df['video_gaze_x'] + offset_x
df['centered_gaze_y'] = df['video_gaze_y'] + offset_y

# Normalize left and right gaze points
df['left_norm_gaze_x'] = (df['left_gaze_x'] - x_min) / (x_max - x_min)
df['left_norm_gaze_y'] = (df['left_gaze_y'] - y_min) / (y_max - y_min)
df['right_norm_gaze_x'] = (df['right_gaze_x'] - x_min) / (x_max - x_min)
df['right_norm_gaze_y'] = (df['right_gaze_y'] - y_min) / (y_max - y_min)

# Map left and right gaze points to video coordinates
df['left_video_gaze_x'] = df['left_norm_gaze_x'] * video_width
df['left_video_gaze_y'] = df['left_norm_gaze_y'] * video_height
df['right_video_gaze_x'] = df['right_norm_gaze_x'] * video_width
df['right_video_gaze_y'] = df['right_norm_gaze_y'] * video_height

# Apply offset to left and right gaze points
df['left_centered_gaze_x'] = df['left_video_gaze_x'] + offset_x
df['left_centered_gaze_y'] = df['left_video_gaze_y'] + offset_y
df['right_centered_gaze_x'] = df['right_video_gaze_x'] + offset_x
df['right_centered_gaze_y'] = df['right_video_gaze_y'] + offset_y

# Function to draw gaze points on a frame
def draw_gaze_points(frame, left_gaze_points, right_gaze_points):
    for point in left_gaze_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < video_width and 0 <= y < video_height:  # Check bounds
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circle for left gaze point
    for point in right_gaze_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < video_width and 0 <= y < video_height:  # Check bounds
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Draw blue circle for right gaze point
    return frame
# Ensure output video resolution matches input video
video_width = frame_width
video_height = frame_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

# Find the last timestamp from the gaze data
last_gaze_timestamp = df['time'].max()
print(f"Last Gaze Timestamp: {last_gaze_timestamp}")

# Function to get the closest gaze data based on timestamp
def get_closest_gaze_data(timestamp):
    if timestamp > last_gaze_timestamp:
        # Extend the last gaze data if timestamp is beyond last gaze timestamp
        closest_idx = df.index[-1]
    else:
        differences = (df['time'] - timestamp).abs()
        closest_idx = differences.idxmin()
    
    left_gaze_points = df.loc[closest_idx, ['left_centered_gaze_x', 'left_centered_gaze_y']].values.reshape(1, -1)
    right_gaze_points = df.loc[closest_idx, ['right_centered_gaze_x', 'right_centered_gaze_y']].values.reshape(1, -1)
    
    return left_gaze_points, right_gaze_points

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_index}")
        break
    if frame is None:
        print(f"Frame {frame_index} is None")
        continue
    
    # Get the timestamp of the current frame in milliseconds
    frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    # Get the closest gaze data based on the current frame timestamp
    left_gaze_points, right_gaze_points = get_closest_gaze_data(frame_timestamp)
    
    # Draw gaze points on the frame
    frame_with_gaze = draw_gaze_points(frame, left_gaze_points, right_gaze_points)
    
    # Write the frame to the output video
    out.write(frame_with_gaze)
    
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved as {output_video_path}")
