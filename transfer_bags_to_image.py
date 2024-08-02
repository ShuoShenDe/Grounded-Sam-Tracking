import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


def save_images_from_rosbag(bag_path, topic_name, output_dir):
    # Create a CvBridge instance
    bridge = CvBridge()
    
    # Open the rosbag
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics = topic_name):
            np_arr = np.fromstring(msg.compressed_data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
            filename = f"{t.to_nsec()}.png"
            cv2.imwrite(os.path.join(output_dir, filename), cv_image)
            print(f"Saved {os.path.join(output_dir, filename)}")

if __name__ == '__main__':
    # Specify the path to your ROS bag file
    bag_path = '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/DS_20240613_101744_6_right.bag'
    topic_name = '/sms_right'  # Make sure to include the space if it's part of the topic name
    output_dir = '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/raw_data/'
    save_images_from_rosbag(bag_path, topic_name, output_dir)