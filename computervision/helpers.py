import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import numpy as np
import keras

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

def my_mse_loss_fn(y_true, y_pred):
    """
    A custom MSE loss function where x and y positions are multiplied by 2,
    but w and h remain the same.
    """
    # Split y_true and y_pred into x, y, w, h components
    x_true, y_true, w_true, h_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x_pred, y_pred, w_pred, h_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    
    # Compute squared differences for each component (x, y, w, h)
    squared_difference_x = keras.ops.square(x_true - x_pred) * 2
    squared_difference_y = keras.ops.square(y_true - y_pred) * 2
    squared_difference_w = keras.ops.square(w_true - w_pred)
    squared_difference_h = keras.ops.square(h_true - h_pred)
    
    # Combine the squared differences (you could use mean or sum depending on your needs)
    total_squared_difference = squared_difference_x + squared_difference_y + squared_difference_w + squared_difference_h
    
    # Return the mean of the squared differences as the loss
    return keras.ops.mean(total_squared_difference)

def iou(y_true, y_pred):
    """
    Calculate IoU loss between the true and predicted bounding boxes.

    y_true and y_pred should have the shape (batch_size, 4), where each element is
    [center_x, center_y, width, height].
    """
    # Convert (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
    true_xmin = y_true[..., 0] - 0.5 * y_true[..., 2]
    true_ymin = y_true[..., 1] - 0.5 * y_true[..., 3]
    true_xmax = y_true[..., 0] + 0.5 * y_true[..., 2]
    true_ymax = y_true[..., 1] + 0.5 * y_true[..., 3]

    pred_xmin = y_pred[..., 0] - 0.5 * y_pred[..., 2]
    pred_ymin = y_pred[..., 1] - 0.5 * y_pred[..., 3]
    pred_xmax = y_pred[..., 0] + 0.5 * y_pred[..., 2]
    pred_ymax = y_pred[..., 1] + 0.5 * y_pred[..., 3]

    # Calculate the intersection area
    inter_xmin = keras.ops.maximum(true_xmin, pred_xmin)
    inter_ymin = keras.ops.maximum(true_ymin, pred_ymin)
    inter_xmax = keras.ops.minimum(true_xmax, pred_xmax)
    inter_ymax = keras.ops.minimum(true_ymax, pred_ymax)

    inter_width = keras.ops.maximum(0.0, inter_xmax - inter_xmin)
    inter_height = keras.ops.maximum(0.0, inter_ymax - inter_ymin)
    intersection_area = inter_width * inter_height

    # Calculate the union area
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union_area = true_area + pred_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

