import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from getCoordinates import get_coordinates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import pandas as pd

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "C:/Users/Computing/Documents/dessertation Code/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)



def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def addPoints(frameIndex=0, object_ID=1):
    base_path = "C:/Users/Computing/Documents/dessertation Code/output_frames"
    ann_frame_idx = frameIndex
    ann_obj_id = object_ID  

    # Format the image path with the index
    image_path = f'{base_path}/{ann_frame_idx:05d}.jpg'

    # Get multiple coordinates
    coordinates = get_coordinates(image_path)
    
    print("Coordinates clicked by the user:")
    print(coordinates)

    # Convert coordinates to numpy array
    coordinates_np = np.array(coordinates, dtype=np.float32)
    
    # Create points array with all coordinates
    points = coordinates_np
    labels = np.ones(len(coordinates_np), dtype=np.int32)  # Assuming all points are positive clicks

    # Ensure each object ID is treated separately in prompts
    if ann_obj_id not in prompts:
        prompts[ann_obj_id] = (points, labels)
    else:
        existing_points, existing_labels = prompts[ann_obj_id]
        prompts[ann_obj_id] = (np.vstack([existing_points, points]), np.hstack([existing_labels, labels]))

    # Perform inference for the current object
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Show the results on the current (interacted) frame
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {ann_frame_idx}")
    
    frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    img = Image.open(frame_path)
    plt.imshow(img)

    show_points(points, labels, plt.gca())

    for i, out_obj_id in enumerate(out_obj_ids):
        if out_obj_id == ann_obj_id:
            show_points(*prompts[out_obj_id], plt.gca())
        
        mask = out_mask_logits[i].cpu().numpy() > 0.0
        show_mask(mask.astype(np.uint8), plt.gca(), obj_id=out_obj_id)

    plt.legend()
    plt.show()

#     return coordinates
def detect_colored_circles(image_path):
    """Detect turquoise and green circles in the given image."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detected_circles = []

    # Define color ranges in HSV
    color_ranges = {
        'turquoise': ((85, 100, 100), (100, 255, 255)),  # HSV range for turquoise
        'green': ((40, 50, 50), (80, 255, 255))          # HSV range for green
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            detected_circles.extend([(circle, color_name) for circle in circles])

    return detected_circles

def check_overlap(circle, mask):
    """Check if the circle overlaps with the given binary mask."""
    cx, cy, r = circle
    mask_height, mask_width = mask.shape
    
    # Create a binary mask for the circle
    circle_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 1, thickness=-1)
    
    # Check overlap
    overlap = np.logical_and(circle_mask, mask)
    return np.any(overlap)

def process_images_for_overlap(image_paths, mask_paths, output_csv='overlap_results.csv'):
    results = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        circles = detect_colored_circles(img_path)
        mask = np.array(Image.open(mask_path).convert('L'))  # Load mask as binary image
        
        img_result = {'image': img_path, 'ROI': 'None', 'Detected_Color': 'None'}
        
        if circles:
            for circle, color in circles:
                if check_overlap(circle, mask):
                    img_result['ROI'] = os.path.basename(mask_path).split('_object_')[1].split('.png')[0]
                    img_result['Detected_Color'] = color
                    break
        
        results.append(img_result)
    
    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Overlap results saved to {output_csv}")
    return df

def get_colormap(num_colors):
    """Generate a colormap with a specified number of distinct colors."""
    colormap = plt.cm.get_cmap('tab20', num_colors)
    return [mcolors.rgb2hex(colormap(i)[:3]) for i in range(num_colors)]

def apply_color_mask(mask, color):
    """Apply a color to a binary mask."""
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask > 0] = color
    return colored_mask

def propagateEntireVideo(output_dir='segmentation_results'):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    all_obj_ids = set()

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: out_mask_logits[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        all_obj_ids.update(out_obj_ids)

    # Generate a colormap for all object IDs
    colormap = get_colormap(len(all_obj_ids))
    obj_id_to_color = {obj_id: tuple(int(colormap[idx].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for idx, obj_id in enumerate(all_obj_ids)}

    # Save each mask separately in the output directory
    for out_frame_idx, masks in video_segments.items():
        for out_obj_id, out_mask in masks.items():
            # Ensure out_mask has the correct dimensions
            mask_shape = out_mask.shape
            out_mask = np.squeeze(out_mask)  # Remove any singleton dimensions

            # Apply the specific color to the mask
            color = obj_id_to_color[out_obj_id]
            colored_mask = apply_color_mask(out_mask, color)

            # Save the colored mask as an image
            mask_filename = f"frame_{out_frame_idx:05d}_object_{out_obj_id}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            mask_image = Image.fromarray(colored_mask)
            mask_image.save(mask_path)

    # Optionally, render the segmentation results every few frames
    vis_frame_stride = 20
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    plt.show()


video_dir = "C:/Users/Computing/Documents/dessertation Code/output_frames"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)
prompts = {}  
addPoints()

while True:
    response = input("Do you want to segment further? (Y/N) or E: ").strip().upper()
        
    if response == 'Y':
        frame_index = int(input("Enter the frame index: ").strip())
        object_ID = int(input("Enter the object ID: ").strip())
        addPoints(frame_index,object_ID)
    elif response == 'N':
        propagateEntireVideo(output_dir='segmentation_results')
        break
    elif response == 'E':
        print("Exiting...")
        break
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")
