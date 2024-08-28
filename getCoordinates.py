import matplotlib.pyplot as plt
import cv2

def get_coordinates(image_path):
    """
    Function to load an image, display it, and capture click coordinates.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - coordinates (tuple): A tuple containing (x, y) coordinates in pixels.
    """
    
    # Load an image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Convert BGR to RGB (because OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # List to store coordinates
    coordinates = []

    def on_click(event):
        # Check if the click is within the image bounds
        if event.xdata is not None and event.ydata is not None:
            # Convert from normalized coordinates to pixel coordinates
            x = int(event.xdata)
            y = int(event.ydata)
            print(f'Coordinates (pixels): ({x}, {y})')

            # Store coordinates
            coordinates.append((x, y))

            # Disconnect the event and close the plot
            fig.canvas.mpl_disconnect(cid)
            plt.close()

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Connect the click event to the on_click function
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    return coordinates if coordinates else None
