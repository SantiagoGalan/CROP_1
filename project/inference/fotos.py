import numpy as np
import matplotlib.pyplot as plt

def foto_mnist(x, size):
    """Reshape a flat vector into a square image (MNIST-style)."""
    return (np.reshape(x, (size, size)) * 255).astype(np.uint8)

def photo_group(images, image_tags, labels=None, num_cols=10, img_size=28):
    """
    Plot images in rows, each row corresponds to a different group.

    Parameters
    ----------
    images : list or np.ndarray
        List of image groups. Each element should be shape (N, img_size*img_size).
        Example: images[0] = group of 10 images from x_mix_orig.
    image_tags : list of str
        Names of each row (group).
    labels : list or None
        Optional labels per group. Not required.
    num_cols : int
        How many images per row.
    img_size : int
        Size of one side of the square image (e.g. 28 for MNIST).
    """
    num_groups = len(images)
    fig, axes = plt.subplots(num_groups, num_cols, figsize=(num_cols, num_groups))

    if num_groups == 1:
        axes = np.expand_dims(axes, 0)  # ensure consistent indexing

    for row in range(num_groups):
        for col in range(num_cols):
            ax = axes[row, col]
            ax.axis("off")

            # Take image from group[row]
            img = foto_mnist(images[row][col], img_size)
            ax.imshow(img, cmap="gray")

            # Add label on the first column only
            if col == 0:
                ax.set_ylabel(image_tags[row], fontsize=10, rotation=0, labelpad=30)

            # Optionally titles for columns (only first row)
            if row == 0:
                ax.set_title(str(col), fontsize=8)

    plt.tight_layout()
    plt.show()
