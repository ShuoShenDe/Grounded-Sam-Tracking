import argparse
import os
import sys
import time

import numpy as np
import json
import torch
from PIL import Image
import gc
from utils.utils import Utils

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
from sam2.build_sam import build_sam2_video_predictor
import copy

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

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

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

def calculate_iou(mask1, mask2):
    # Convert masks to float tensors for calculations
    mask1 = mask1.to(torch.float32)
    mask2 = mask2.to(torch.float32)
    
    # Calculate intersection and union
    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - intersection
    
    # Calculate IoU
    iou = intersection / union
    return iou


def save_mask_data2(output_dir, mask_list, box_list, label_list, output_file_name="mask.npy", value = 0 ): # 0 for background
    mask_img = torch.zeros(mask_list.shape[-2:])
    anno_2d = []
    for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
        final_index = value + idx + 1
        mask_img[mask[0] == True] = final_index
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        box = box.numpy().tolist()
        if box[3] > 1077:
            continue
        else:
            anno_2d.append({
                'instance_id': final_index,
                'class_name': name,
                'class_score': float(logit),
                'x1': box[0],
                'y1': box[1],
                'x2': box[2],
                'y2': box[3],
                'mask': mask
            })

    # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))

    json_data = {
        "image_name": output_file_name,
        "image_height": mask_img.shape[0],
        "image_width": mask_img.shape[1],
        "anno_2d": anno_2d
    }

    # with open(os.path.join(output_dir, f"{os.path.splitext(output_file_name)[0]}.json"), 'w') as json_file:
    #     json.dump(json_data, json_file, indent=4)
    # with open(os.path.join(output_dir, output_file_name.split(".")[0]+'.json'), 'w') as f:
    #     json.dump(json_data, f)
    return json_data

def sam2_init(video_dir):
    torch.autocast(device_type="cuda").__enter__()  # , dtype=torch.bfloat16

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_checkpoint = "segment_anything_2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=video_dir)
    return predictor, inference_state

def get_new_box(mask):
    # 找到所有非零值的索引
    nonzero_indices = torch.nonzero(mask)
    
    # 如果没有非零值，返回一个空的边界框
    if nonzero_indices.size(0) == 0:
        # print("nonzero_indices", nonzero_indices)
        return []
    
    # 计算最小和最大索引
    y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
    y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
    
    # 创建边界框 [x_min, y_min, x_max, y_max]
    bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
    
    return bbox

def sam_init():
    # initialize SAM
    sam_version = "vit_h"
    sam_checkpoint = "./sam_hq_vit.pth"
    sam_hq_checkpoint = "./sam_hq_vit_h.pth"
    use_sam_hq = True
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    return predictor


def sam2_tracking(predictor,inference_state, masks_dict_list, start_frame_idx=0, max_frame_num_to_track=10):
    predictor.reset_state(inference_state)
    for mask_dict in masks_dict_list:
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                mask_dict["instance_id"],
                mask_dict["mask"][0],
            )
    video_segments = {} 
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, max_frame_num_to_track=max_frame_num_to_track, start_frame_idx=start_frame_idx):
        anno_2d = []
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            
            target_mask_dictionary = get_annotation_dictionary(masks_dict_list, target_instance_id=out_obj_id)
            target_mask_dictionary["mask"] = out_mask
            anno_2d.append(target_mask_dictionary)

        video_segments[out_frame_idx] = anno_2d
    return video_segments

def get_annotation_dictionary(anno_2d, target_instance_id=None):
    for anno in anno_2d:
        if anno["instance_id"] - target_instance_id == 0:
            new_dict = copy.deepcopy(anno)
            return new_dict
    return {}

def get_tracking_masks(previous_track_masks, new_seg_masks, objects_count=0):
    final_tracking_masks = []
    for new_seg_mask in new_seg_masks:
        flag = False
        
        new_seg_mask_copy = copy.deepcopy(new_seg_mask)
        for previous_track_mask in previous_track_masks:
            iou = calculate_iou(new_seg_mask['mask'], previous_track_mask['mask'])  # tensor, numpy
            # print("iou", iou)
            if iou > 0.8:
                flag = True
                new_seg_mask_copy["instance_id"] = previous_track_mask["instance_id"]
                final_tracking_masks.append(new_seg_mask_copy)
                break
        if not flag:
            objects_count += 1
            new_seg_mask_copy['instance_id'] = objects_count
            final_tracking_masks.append(new_seg_mask_copy)
    return final_tracking_masks, objects_count

def save_video_masks(output_path, video_masks, image_name_batch):

    for frame_id, mask_list in video_masks.items():
        image_base_name = image_name_batch[frame_id].split(".")[0]
        json_name = f"mask_{image_base_name}.json"
        mask_name = f"mask_{image_base_name}.npy"
        json_data = {
            "image_name": mask_name,
            "image_height": 1080,
            "image_width": 1920,
        }
        anno_2d = []
        try:
            mask = mask_list[0]["mask"]
            mask_img = torch.zeros(mask.shape[-2:])
        except:
            mask_img = torch.zeros((1080, 1920))
        for mask_dict in mask_list:
            mask = mask_dict["mask"]  # .cpu().numpy()
            mask_img[mask[0] == True] = mask_dict["instance_id"]
            
            deep_copied_dict = copy.deepcopy(mask_dict)
            deep_copied_dict.pop("mask")
            if " " in deep_copied_dict["class_name"]:
                deep_copied_dict["class_name"] = deep_copied_dict["class_name"].split(" ")[0]
            box = get_new_box(mask[0])
            if box:
                deep_copied_dict["x1"] = box[0]
                deep_copied_dict["y1"] = box[1]
                deep_copied_dict["x2"] = box[2]
                deep_copied_dict["y2"] = box[3]
                anno_2d.append(deep_copied_dict)
        np.save(os.path.join(output_path, "mask_data" ,mask_name), mask_img.numpy().astype(np.uint16))
        json_data["anno_2d"] = anno_2d
        with open(os.path.join(output_path,"json_data", json_name), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


def get_last_video_mask(video_masks):
    last_frame_ids = list(video_masks.keys())
    last_frame_ids.sort()
    last_frame_id = last_frame_ids[-1]
    print(video_masks.keys(), "get last_frame_id", last_frame_id)
    return video_masks[last_frame_id]


# at 3090 max load 80 images  # 50 images 9900MiB
if __name__ == "__main__":   
    # ************  Parameters  ************
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Batch", add_help=True)
    parser.add_argument("--box_threshold", type=float, default=0.2, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
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
        input_dir = "/media/NAS/sd_nas_01/shuo/denso_data/trip_full/sms_front"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "/media/NAS/sd_nas_01/shuo/denso_data/trip_full/"

    # cfg
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "groundingdino_swint_ogc.pth"  # change the path of the model

    text_prompt = "car.pole.van.pedestrian."  # c
    
    device = "cuda"
    # 每10个执行一次
    step = 10
    objects_count = 0


    print("box_threshold", box_threshold, "text_threshold", text_threshold)
    print("input_dir", input_dir,"\n", "output_dir", output_dir)    
    # ************  initialize  ************
    Utils.creat_dirs(output_dir)
    Utils.creat_dirs(os.path.join(output_dir, "mask_data"))
    Utils.creat_dirs(os.path.join(output_dir, "json_data"))

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    sam2_tracking_predictor, inference_state = sam2_init(input_dir)
    sam_seg_predictor = sam_init()
    # load image
    image_names = os.listdir(input_dir)
    image_names.sort()
    total_iterations = len(image_names)
    tracking_masks = []
    start_time = time.time()

    for start_frame_idx in range(0, total_iterations, step):
        image_name = image_names[start_frame_idx]
        image_path = os.path.join(input_dir, image_name)
        image_pil, image = load_image(image_path)
        image_full_name = image_name.split(".")[0]
        image_appendix = image_name.split(".")[1]
        # visualize raw image

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        print(pred_phrases)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_seg_predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_seg_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        if transformed_boxes.size(0) == 0:
            print("{} frame {} nothing recognized".format(start_frame_idx, image_full_name))
            continue
        masks, _, _ = sam_seg_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        
        # draw output image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, "grounded_sam_output_{}.jpg".format(image_full_name)),
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )
        # plt.close('all')  


        # *******  run sam2_tracking model ****************
        new_seg_masks = save_mask_data2(output_dir, masks, boxes_filt, pred_phrases, output_file_name=f"mask_{str(image_full_name)}.npy")
        
        sam2_tracking_masks, objects_count = get_tracking_masks(tracking_masks, new_seg_masks['anno_2d'], objects_count)
        video_segments = sam2_tracking(sam2_tracking_predictor, inference_state, sam2_tracking_masks, start_frame_idx=start_frame_idx)
        tracking_masks = get_last_video_mask(video_segments) 
        save_video_masks(output_dir, video_segments, image_names)

        print(start_frame_idx,"Total Time taken: ", time.time() - start_time)
        print("total objects count {} in during frame {}-{}".format(objects_count, start_frame_idx, start_frame_idx+step))
        if start_frame_idx%10 == 0:
            # Call the garbage collector
            gc.collect()
            # Empty the PyTorch cache
            torch.cuda.empty_cache()
            print("cleaning")
        
