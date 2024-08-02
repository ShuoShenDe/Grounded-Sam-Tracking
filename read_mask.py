import numpy as np
import matplotlib.pyplot as plt

def load_mask_data(filepath):
    mask_img = np.load(filepath)

    return mask_img

def count_values(array, value):
    return np.sum(array == value)
# read mask data
mask_filepath = '/media/NAS/sd_nas_01/shuo/denso_data/pretrain_clean/test_short_data_mask/front/mask_1718266742040110000.npy'  
mask_img = load_mask_data(mask_filepath)
print(mask_img[0], type(mask_img[0]))
print(np.unique(mask_img).tolist())
print(mask_img.shape)

count_3 = count_values(mask_img, 3)
print("count_3", count_3)
count_4 = count_values(mask_img, 4)
print("count_4", count_4)

# show final images
plt.figure(figsize=(10, 10))
plt.imshow(mask_img[0], cmap='gray')
plt.axis('off')
plt.savefig("test.jpg", bbox_inches="tight", dpi=300, pad_inches=0.0)