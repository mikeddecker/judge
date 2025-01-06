import matplotlib.pyplot as plt
import cv2

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            # Ensure img is in the correct format (BGR to RGB)
            if img.shape[-1] == 3:  # If the image has 3 channels (likely BGR from OpenCV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            ax = axs[row_idx, col_idx]
            ax.imshow(img, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
