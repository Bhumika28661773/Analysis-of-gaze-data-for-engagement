import cv2
import os

# Path to the video file
video_path = 'croppedVideo.mp4'

# Directory where the frames will be saved
output_folder = 'output_frames'
os.makedirs(output_folder, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)

# Frame count
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Generate the frame file name
    frame_filename = os.path.join(output_folder, f'{frame_count:05d}.jpg')
    
    # Save the frame as an image
    cv2.imwrite(frame_filename, frame)
    
    # Increment the frame count
    frame_count += 1

# Release the video capture object
cap.release()

print(f"Extracted {frame_count} frames from the video and saved in {output_folder}.")
