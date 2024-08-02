import os
import shutil
import numpy as np

class Utils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")


    def split_trip_to_small(input_dir, output_dir, trip_length=50):
        """
        Split a long trip into smaller trips.
        """
        images_name_list = os.listdir(input_dir)
        images_name_list.sort()
        trip = []
        trip_folder = []
        for i in range(0,len(images_name_list),trip_length):
            trip = images_name_list[i:i+trip_length]
            trip_dir = os.path.join(output_dir, str(i))
            Utils.creat_dirs(trip_dir)
            for image in trip:
                image_path = os.path.join(input_dir, image)
                target_image_path = os.path.join(trip_dir, image)
                if not os.path.exists(target_image_path):
                    shutil.copy(os.path.join(input_dir, image), trip_dir)
            trip_folder.append(str(i))
        
        return trip_folder