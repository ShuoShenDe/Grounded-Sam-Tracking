# prepare
prepare the images data as the follow rules:
/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/raw_data

where sms_front is sensor name, raw_data is fixed

***note***
Now the data is located under the path: /media/NAS/sd_nas_01/shuo/denso_data/ (because this is the only volume of the container ca9bb3230d4e)

# Step 1 : Open the env
Now it only works on 8090

```
docker exec -it ca9bb3230d4e /bin/bash

cd /home/appuser/Grounded-Segment-Anything
```
# Step 2: Run Code
```
python run_long_trip_to_tracking.py 
```

# What you can change in the code

*** 1. input_dir ***
Please change `input_dir`,   `output_dir`, `trip_length`, `box_threshold`
```
    input_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/raw_data"
    output_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/"
    trip_length = 50
    box_threshold = 0.23

```

Then the code will automatically run the following command:
```
python grounded_sam_with_sam_tracking.py -i /media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/0 -o /media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/ --box_threshold 0.23

```

If you wan to see the result of pretraining, please run:
```
python draw_raw_image_and_box.py
```



