from utils.utils import Utils
import subprocess
import os
import shutil

if __name__ == "__main__":
    input_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/raw_data"
    output_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/"
    trip_length = 50
    box_threshold = 0.23


    trip_count = Utils.split_trip_to_small(input_dir, output_dir, trip_length=trip_length)
    
    for i_trip_count in trip_count:
        input_tracking_raw_image = os.path.join(output_dir, str(i_trip_count))
        command = "python grounded_sam_with_sam_tracking.py -i {} -o {} --box_threshold {}".format(input_tracking_raw_image, output_dir, box_threshold)
        # 使用 subprocess.run 运行命令
        print(command)
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        # 输出命令的输出
        print("Output:", result.stdout)