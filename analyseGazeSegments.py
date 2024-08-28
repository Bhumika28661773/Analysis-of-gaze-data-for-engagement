import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image

def load_image(image_path):
    """Load an image from a file path."""
    return cv2.imread(image_path)

def load_segmented_masks(segmented_folder, frame_name, img_shape):
    """Load segmented masks for a given frame and resize them to match the image size."""
    masks = []
    for obj_id in [1, 2]:  # Assuming object IDs are 1 and 2
        # Create the path with forward slashes
        mask_path = os.path.join(segmented_folder, f"frame_{frame_name}_object_{obj_id}.png").replace('\\', '/')
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))  # Load mask as grayscale
            mask_resized = cv2.resize(mask, (img_shape[1], img_shape[0]))  # Resize mask to match image dimensions
            masks.append(mask_resized)
        else:
            print("Mask not found:", mask_path)  # Debugging: Print if mask is not found

    return masks

def detect_colored_spots(image_path):
    """Detect green and blue spots in the given image."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detected_spots = []

    # Define color ranges in HSV
    color_ranges = {
        'blue': ((110, 150, 50), (130, 255, 255)),   # HSV range for pure blue #0000FF
        'green': ((50, 100, 100), (70, 255, 255))    # HSV range for pure green #00FF00
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # Find contours of the spots
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                detected_spots.append((x, y, w, h, color_name))

    return detected_spots

def check_overlap(spot, masks, img_shape):
    """Check overlap between a detected spot and a list of masks."""
    x, y, w, h, color = spot
    spot_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.rectangle(spot_mask, (x, y), (x+w, y+h), 255, -1)
    overlap_info = []
    for obj_id, mask in enumerate(masks, start=1):
        overlap = np.logical_and(spot_mask > 0, mask > 0)
        if np.sum(overlap) > 0:
            overlap_info.append(obj_id)

    return overlap_info, spot_mask

def visualize_results(image, detected_spots, masks, overlaps, output_path):
    """Visualize the detected spots and their overlaps with segmented masks."""
    # Draw detected spots
    for (x, y, w, h, color) in detected_spots:
        spot_color = (0, 255, 0) if color == 'green' else (255, 0, 0)  # Green or Blue
        cv2.rectangle(image, (x, y), (x+w, y+h), spot_color, 2)
        cv2.putText(image, f'{color.capitalize()} Spot', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spot_color, 2)

    # Draw masks
    for obj_id, mask in enumerate(masks, start=1):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 0, 0), 2)  # Draw contours in blue

    # Overlay spot masks with transparency
    for detected_colors, spot_mask in overlaps:
        alpha = 0.5  # Transparency factor
        overlay = image.copy()
        overlay[spot_mask > 0] = (0, 255, 255)  # Yellow for overlap areas
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Save the result
    cv2.imwrite(output_path, image)

def process_images(image_folder, segmented_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    results = []
    image_results = {}  # Dictionary to keep track of best results

    for image_file in image_files:
        frame_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_folder, image_file)

        # Load image to get its dimensions
        img = load_image(image_path)
        img_shape = img.shape[:2]  # (height, width)

        # Detect spots in the image
        spots = detect_colored_spots(image_path)

        # Load segmented masks for the current frame
        masks = load_segmented_masks(segmented_folder, frame_name, img_shape)
        overlaps = []

        has_non_none_roi = False
        for spot in spots:
            x, y, w, h, color = spot
            overlap_info, spot_mask = check_overlap((x, y, w, h, color), masks, img_shape)
            if overlap_info:
                detected_colors = [f'Object {obj_id}' for obj_id in overlap_info]
                overlaps.append((detected_colors, spot_mask))
                roi_info = ', '.join(detected_colors)
                has_non_none_roi = True
            else:
                roi_info = 'Background'

            # Update the dictionary only if we don't have a better ROI already
            if image_file not in image_results or (roi_info != 'None' and not has_non_none_roi):
                image_results[image_file] = {
                    'Image': image_file,
                    'ROI': roi_info
                }

        # Visualize and save the result
        output_path = os.path.join(output_folder, f"{frame_name}_result.png")
        visualize_results(img, spots, masks, overlaps, output_path)

    # Convert the dictionary to a DataFrame
    results = list(image_results.values())
    return results

# Example usage
image_folder = "C:/Users/Computing/Documents/dessertation Code/output_frames"
segmented_folder = "C:/Users/Computing/Documents/dessertation Code/segmentation_results"
output_folder = 'output_folder'

results = process_images(image_folder, segmented_folder, output_folder)

# Convert results to a DataFrame and save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('detection_results.csv', index=False)
print(df_results)
