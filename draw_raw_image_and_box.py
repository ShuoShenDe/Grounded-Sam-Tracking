import json
import os
import cv2
import numpy as np
import random
from utils.utils import Utils

def random_color():
    """生成一个随机颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


root_path = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front"  # Adjust the path to your dataset directory

json_path = os.path.join(root_path, "json_data")  # Adjust the path to your JSON file
mask_path = os.path.join(root_path, "mask_data")  # Adjust the path to your mask file
raw_image_path = os.path.join(root_path, "raw_data")  # Adjust the path to your raw image file
output_path = os.path.join(root_path, "output_image")  # Adjust the path to your output image file
Utils.creat_dirs(output_path)
raw_image_name_list = os.listdir(raw_image_path)
raw_image_name_list.sort()
for raw_image_name in raw_image_name_list:
    image_path = os.path.join(raw_image_path, raw_image_name)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image file not found.")
    # load mask
    mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
    mask = np.load(mask_npy_path)
    # color map
    unique_ids = np.unique(mask)
    colors = {uid: random_color() for uid in unique_ids}
    # apply mask to image
    colored_mask = np.zeros_like(image)
    for uid in unique_ids:
        colored_mask[mask == uid] = colors[uid]
    alpha = 0.5  # 调整 alpha 值以改变透明度
    output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


    file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        # Draw bounding boxes and labels
        for item in json_data["anno_2d"]:
            # Extract data from JSON
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            instance_id = item["instance_id"]
            class_name = item["class_name"]

            # Draw rectangle
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put text
            label = f"{instance_id}: {class_name}"
            cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save the modified image
        output_image_path = os.path.join(output_path, raw_image_name)
        cv2.imwrite(output_image_path, output_image)

        print(f"Annotated image saved as {output_image_path}")