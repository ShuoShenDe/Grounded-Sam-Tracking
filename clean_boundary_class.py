import os
import json
import numpy as np

def get_all_json_files(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.json'):
                json_files.append(os.path.join(dirpath, file))
    return json_files

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def clean_boundary_class():
    root_path = "/media/NAS/sd_nas_01/shuo/denso_data/tracking/"
    key_word = "/tracking"
    sensor_names = os.listdir(root_path)

    for sensor_name in sensor_names:
        json_root = os.path.join(root_path, sensor_name)
        json_files = get_all_json_files(json_root)
        
        for json_file in json_files:
            data = read_json_file(json_file)
            image_path = json_file.replace(".json", ".npy")
            new_json_path = json_file.replace(key_word, key_word+"_clean")
            new_img_path = image_path.replace(key_word, key_word+"_clean")

            
            img_array = np.load(image_path)
            new_json = {
                "image_name": data["image_name"],
                "image_height": data["image_height"],
                "image_width": data["image_width"],
            }

            anno_2d = []
            
            remove_values = []
            for box in data["anno_2d"]:
                if box["instance_id"] == "0":
                    continue
                elif box["y2"] > 1077:
                    value = box["instance_id"]
                    remove_values.append(np.dtype(img_array.dtype).type(value))
                else:
                    if " " in box["class_name"]:
                        box["class_name"] = box["class_name"].split(" ")[0]
                    print("type(box['instance_id'])", type(box["instance_id"]))
                    anno_2d.append(box)
            
            new_json["anno_2d"] = anno_2d
            
            new_json_path_root = os.path.dirname(new_json_path)
            
            if not os.path.exists(new_json_path_root):
                os.makedirs(new_json_path_root)
            
            with open(new_json_path, 'w') as f:
                json.dump(new_json, f)
            
            print(json_file, "remove_values", remove_values)
            
            img_array[np.isin(img_array, remove_values)] = 0
            np.save(new_img_path, img_array.astype(np.uint16))

if __name__ == "__main__":
    clean_boundary_class()