import numpy as np
import matplotlib.pyplot as plt

def load_mask_data(filepath):
    mask_img = np.load(filepath)

    return mask_img

# read mask data
mask_filepath = 'mask_1718267167846453000.npy'  
mask_img = load_mask_data(mask_filepath)
print(mask_img[0], type(mask_img[0]))


# show final images
plt.figure(figsize=(10, 10))
plt.imshow(mask_img, cmap='gray')
plt.axis('off')
plt.savefig("test.jpg", bbox_inches="tight", dpi=300, pad_inches=0.0)