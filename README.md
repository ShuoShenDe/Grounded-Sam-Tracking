# Why built this project

I created a project with the purpose of using Grounded-DINO, SAM (Segment Anything Model), and tracking algorithms to achieve text-prompt-based object recognition and continuous tracking in videos. This combination allows for precise and efficient identification and tracking of objects within video content based on textual descriptions.

Objectives:

	1.	Text-Prompt Based Object Recognition:
	•	Utilize Grounded-DINO to interpret and understand textual prompts for object identification within video frames.
	2.	Segmentation and Analysis:
	•	Implement SAM (Segment Anything Model) to accurately segment and analyze objects in video frames based on the prompts provided.
	3.	Continuous Object Tracking:
	•	Apply sam2 tracking algorithms to maintain and follow the identified objects throughout the video, ensuring consistent and reliable tracking over time.

Benefits:

	•	Efficiency: Streamline the process of object recognition and tracking by leveraging state-of-the-art models.
	•	Accuracy: Enhance the precision of object identification and tracking through advanced segmentation techniques.
	•	Automation: Enable automated monitoring and analysis of video content based on textual descriptions, reducing the need for manual intervention.

This project aims to integrate cutting-edge technologies in computer vision and natural language processing to create a robust system for video content analysis and tracking.


# prepare
prepare the images data as the follow rules:
/your_own_path/raw_data

where raw_data is fixed

***note***
Now the data is located under the path: /media/NAS/sd_nas_01/shuo/denso_data/ (because this is the only volume of the container ca9bb3230d4e)

# Step 1 : Open the env
You could use the dockerfile, and create environment as [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything.git)

``

cd /home/appuser/Grounded-Segment-Anything
```
# Step 2: Run Code

Then command example:
```
python grounded_sam_with_sam_tracking.py -i /your_path_to_data/raw_data -o /your_path_to_data/ --box_threshold 0.23

```

If you wan to see the result of pretraining, please run:
```
python draw_raw_image_and_box.py
```

# 💘 Acknowledgements
[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything.git)
[Segment-anything-2](https://github.com/facebookresearch/segment-anything-2.git)



