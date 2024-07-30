import os
import glob
import json
from PIL import Image
import numpy as np


def get_all_json_files(path):
    # Get all JSON files under the specified path
    json_files = glob.glob(os.path.join(path, '*.json'))
    return json_files

def read_json_file(file_path):
    # Read a JSON file and return its content
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def clean_boundary_class():
    root_path = "/home/ubuntu/Documents/EFS/Labeling/Denso/pretrain/20240613_101744_6"

    sensor_names = os.listdir(root_path)

    for sensor_name in sensor_names:
        json_root = os.path.join(root_path, sensor_name)
        json_files = get_all_json_files(json_root)
        
        for json_file in json_files:
            image_path = json_file.replace(".json", ".npy")
            new_json = {
                "image_name": os.path.basename(image_path),
                "image_height": 1080,
                "image_width": 1920,
            }

            anno_2d = []
            data = read_json_file(json_file)
            remove_values = []
            for box in data["anno_2d"]:
                if box["instance_id"] == "0":
                    anno_2d.append(box)
                    continue
                elif box["y2"] > 1077:
                    value = box["instance_id"]
                    remove_values.append(value)
                else:
                    anno_2d.append(box)
            new_json["anno_2d"] = anno_2d
            new_json_path = json_file.replace("/pretrain/", "/pretrain_clean/")
            # print(new_json_path, new_json)
            # print(json_file, remove_value)
            new_json_path_root = os.path.dirname(new_json_path)
            if not os.path.exists(new_json_path_root):
                print("create data")
                os.makedirs(new_json_path_root)
            with open(new_json_path, 'w') as f:
                json.dump(new_json, f)            
            for remove_value in remove_values:
                # Open the image
                new_img_path = image_path.replace("/pretrain/", "/pretrain_clean/")
                img_array = np.load(image_path)
                # Convert the image to a numpy array
                remove_value = np.dtype(img_array.dtype).type(remove_value)
                print("remove_value", remove_value)
                img_array[img_array == remove_value] = 0
                np.save(new_img_path, img_array.astype(np.uint16))
                    # print('Saved modified image to {}'.format(new_img_path))
        
       #  break



if __name__ == "__main__":
    clean_boundary_class()