import fast
import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = "/Users/andreped/workspace/BCSS/data/"

for image_id in os.listdir(data_dir + "images/"):
    image_path = data_dir + "images/" + image_id
    mask_path = data_dir + "masks/" + image_id

    image = plt.imread(image_path)
    mask = plt.imread(mask_path)
    mask = (mask * 255).astype("uint8")

    fast_image = fast.Image.createFromArray(image)
    fast_mask = fast.Image.createFromArray(mask)

    generators = [fast.PatchGenerator.create(512, 512).connect(0, curr) for curr in [fast_image, fast_mask]]
    streamers = [fast.DataStream(curr) for curr in generators]

    for image_patch, mask_patch in zip(*streamers):

        image_patch = np.asarray(image_patch).astype("float32")
        mask_patch = np.asarray(mask_patch).astype("uint8")

        image_patch = image_patch / np.amax(image_patch)

        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        ax[0].imshow(image_patch)
        ax[1].imshow(mask_patch, cmap="jet", interpolation="none")
        ax[2].imshow(image_patch)
        ax[2].imshow(mask_patch, cmap="jet", interpolation="none", alpha=0.5)
        for i in range(3):
            ax[i].axis("off")
        fig.tight_layout()
        plt.show()
