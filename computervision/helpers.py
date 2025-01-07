import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import numpy as np

def plot(imgs, bboxes=None, row_title=None, **imshow_kwargs):
    """
    Plot images in a grid with optional bounding boxes.

    imgs: list of images or list of lists of images (in BGR format).
    bboxes: list of bounding boxes corresponding to each image.
            Each bounding box is in the format [center_x, center_y, width, height], normalized (0 to 1).
            If None, no bounding boxes are drawn.
    row_title: optional list of titles for each row.
    **imshow_kwargs: additional arguments passed to `imshow`.
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = 2  # 2 images per row

    # Adjust the figure size to make the images bigger and fit 2 images per row
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(12, 6))
    
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            # Ensure img is in the correct format (BGR to RGB)
            if img.shape[-1] == 3:  # If the image has 3 channels (likely BGR from OpenCV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            ax = axs[row_idx, col_idx]
            ax.imshow(img, **imshow_kwargs)
            
            # If bounding boxes are provided, draw them on the image
            if bboxes is not None:
                bbox = bboxes[row_idx * num_cols + col_idx]  # Get the bounding box for the current image
                if bbox is not None:
                    # Unpack the bounding box (center_x, center_y, width, height)
                    center_x, center_y, width, height = bbox

                    # Get the image dimensions
                    img_height, img_width = img.shape[:2]

                    # Convert the normalized bounding box to pixel coordinates
                    xmin = (center_x - width / 2) * img_width
                    ymin = (center_y - height / 2) * img_height
                    xmax = (center_x + width / 2) * img_width
                    ymax = (center_y + height / 2) * img_height

                    # Create a rectangle patch for the bounding box
                    rect = patches.Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)  # Add the bounding box to the image
            ax.axis('off')

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    # No need for tight_layout() to avoid any unwanted resizing of images
    plt.show()
