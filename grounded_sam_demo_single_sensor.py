import argparse
import os
import sys
import time

import numpy as np
import json
import torch
from PIL import Image
import gc

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)



def save_mask_data2(output_dir, mask_list, box_list, label_list, output_file_name="mask.jpg"):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    # plt.savefig(os.path.join(output_dir, output_file_name), bbox_inches="tight", dpi=300, pad_inches=0.0)
    np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
    json_data = {
        "image_name": output_file_name,
        "image_height":mask_img.shape[0],
        "image_width":mask_img.shape[1]
    }

    anno_2d = [{
        'instance_id': str(value),
        'class_name': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        box = box.numpy().tolist()
        anno_2d.append({
            'instance_id': str(value),
            'class_name': name,
            'class_score': float(logit),
            'x1': box[0],
            'y1': box[1],
            'x2': box[2],
            'y2': box[3]
        })
    json_data["anno_2d"] = anno_2d
    with open(os.path.join(output_dir, output_file_name.split(".")[0]+'.json'), 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Batch", add_help=True)
    parser.add_argument("--box_threshold", type=float, default=0.2, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.3, help="text threshold")
    parser.add_argument(
        "--output_dir", "-o", type=str, help="output directory"
    )
    parser.add_argument("--input_dir", "-i", type=str,  help="path to image file")

    args = parser.parse_args()
    if args.box_threshold:
        box_threshold = args.box_threshold

    if args.text_threshold:
        text_threshold = args.text_threshold

    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = "/media/NAS/sd_nas_01/shuo/denso_data/test_front/test_short_data"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "/media/NAS/sd_nas_01/shuo/denso_data/test_front/test_short_data_mask"

    print("box_threshold", box_threshold, "text_threshold", text_threshold)
    print("input_dir", input_dir, "output_dir", output_dir)
    start_time = time.time()
    # cfg
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "groundingdino_swint_ogc.pth"  # change the path of the model
    sam_version = "vit_h"
    sam_checkpoint = "./sam_hq_vit.pth"
    sam_hq_checkpoint = "./sam_hq_vit_h.pth"
    use_sam_hq = True
    
    text_prompt = "car.van.pedestrian.pole."
    
    device = "cuda"
    
    
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    # load image

    for index, image_name in enumerate(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, image_name)
        image_pil, image = load_image(image_path)
        image_full_name = image_name.split(".")[0]
        image_appendix = image_name.split(".")[1]
        # visualize raw image
        # image_pil.save(os.path.join(output_path, "{}_{}_{}.{}".format(image_full_name, "_raw",str(index),image_appendix)))

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        print(pred_phrases)
        print(len(boxes_filt))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        # print("transformed_boxes", transformed_boxes.shape)
        if transformed_boxes.size(0) == 0:
            print("{} frame {} nothing recognized".format(index, image_full_name))
            continue
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output_{}.jpg".format(image_full_name)),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        save_mask_data2(output_dir, masks, boxes_filt, pred_phrases, output_file_name=f"mask_{str(image_full_name)}.npy")
        print(index,"Total Time taken: ", time.time() - start_time)
        plt.close('all')  # 关闭所有的figure以释放内存

        if index%10 == 0:
            # Call the garbage collector
            gc.collect()
            # Empty the PyTorch cache
            torch.cuda.empty_cache()
            print("cleaning")
