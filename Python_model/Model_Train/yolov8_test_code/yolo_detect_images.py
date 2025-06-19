import os
import cv2
from yoloraw_postprocessing import *

model_path = r'C:\Users\lexho\Documents\GitHub\TWKibo6th\Python_model\Model_Train\yolov8_weight\krpc_aug_yolov8n_32b_cos_lr_no_rot_with03_mixup_copy_paste\weights\best.pt'
image_base_folder = r'C:\Users\lexho\Documents\GitHub\TWKibo6th\Python_model\Model_Train\yolov8_test_code'
image_names = ['area_2_yolo_original_320x320_enhanced.png']#, 'yolo_binary_otsu_2.png']#, '0080.png', '0021.png', '0157.png']
img_type = "lost"  # or "target"
img_size = 320
conf_threshold = 0.3
standard_nms_threshold = 0.45
overlap_nms_threshold = 0.8

print(f"\n{'='*60}")
# Create cv_img_list from base folder and image names
cv_img_list = []
for image_name in image_names:
    image_path = os.path.join(image_base_folder, image_name)
    cv_img = load_image_path(image_path, img_size=img_size)
    cv_img_list.append(cv_img)

# Option 1: Simple detection (recommended for actual use)
detections = simple_detection_example(
    model_path=model_path, 
    cv_img_list=cv_img_list, 
    img_type=img_type,
    img_size=img_size, 
    conf_threshold=conf_threshold, 
    standard_nms_threshold=standard_nms_threshold, 
    overlap_nms_threshold=overlap_nms_threshold
)
print(f"\nFinal detection list: {detections}")

report_landmark = []
store_treasure = []
for i, detection in enumerate(detections):
    report_landmark.append(detection['landmark_quantities'])
    store_treasure.append(detection['treasure_quantities'].keys())
print(f"\nReport landmark quantities: {report_landmark}")
print(f"Store treasure quantities: {store_treasure}")


# Option 2: Test different parameters (for experimentation)
# test_pipeline_with_different_params()