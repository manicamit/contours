#!/usr/bin/env python3
# Updated Git configuration test
import argparse
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image
import random
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Trace object outlines in an image using Mask R-CNN")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", help="Path to save the output image")
    parser.add_argument("--thickness", type=int, default=2, help="Thickness of the outline")
    parser.add_argument("--color", default="random", help="Color of the outline: random, white, red, green, blue, or R,G,B values")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--classes", help="Filter by classes, comma-separated")
    parser.add_argument("--show-labels", action="store_true", help="Display class names and confidence scores")
    return parser.parse_args()

def get_color(color_arg):
    if color_arg == "random":
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    elif color_arg == "white":
        return (255, 255, 255)
    elif color_arg == "red":
        return (0, 0, 255)  # BGR format for OpenCV
    elif color_arg == "green":
        return (0, 255, 0)
    elif color_arg == "blue":
        return (255, 0, 0)
    else:
        try:
            # Parse R,G,B values
            r, g, b = map(int, color_arg.split(','))
            return (b, g, r)  # BGR format for OpenCV
        except:
            print(f"Invalid color format: {color_arg}. Using random color.")
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def load_model():
    # Load the pre-trained Mask R-CNN v2 model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, device

def get_coco_class_names():
    # COCO dataset class names
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return COCO_CLASSES

def main():
    args = parse_args()
    
    # Load the image
    image_path = args.image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert the image to RGB (from BGR) for the model
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Load the Mask R-CNN v2 model
    model, device = load_model()
    
    # Preprocess the image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Get class names
    coco_classes = get_coco_class_names()
    
    # Filter classes if specified
    target_classes = None
    if args.classes:
        target_classes = [cls.strip().lower() for cls in args.classes.split(',')]
    
    # Get the outline color
    color = get_color(args.color)
    
    print("Processing image with Mask R-CNN ResNet50 FPN v2...")
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_batch)
    
    # Process results
    masks = predictions[0]['masks'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Create a copy of the original image for drawing
    result_image = original_image.copy()
    
    detection_count = 0
    
    # Process each detection
    for i in range(len(scores)):
        if scores[i] >= args.conf:
            class_id = labels[i]
            class_name = coco_classes[class_id]
            
            # Skip if not in target classes
            if target_classes and class_name.lower() not in target_classes:
                continue
            
            # Get the mask
            mask = masks[i, 0]  # First channel of the mask
            mask = (mask > 0.5).astype(np.uint8)  # Threshold the mask
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the contours
            cv2.drawContours(result_image, contours, -1, color, args.thickness)
            
            # Add label if enabled
            if args.show_labels:
                # Find the top-most point in the contour to place the label
                if contours:
                    # Find a good position for the label (top of the contour)
                    min_y = min([pt[0][1] for pt in contours[0]])
                    x = contours[0][0][0][0]  # Use x from the first point
                    y = max(min_y - 10, 10)  # Place text above the contour with padding
                    
                    # Create label text with class name and confidence
                    label_text = f"{class_name}: {scores[i]:.2f}"
                    
                    # Get text size for background rectangle
                    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # Draw background rectangle
                    cv2.rectangle(result_image, 
                                (x, y - text_size[1] - 5),
                                (x + text_size[0] + 5, y + 5),
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(result_image, label_text, (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            detection_count += 1
            
            # Use a different color for each object if random color is chosen
            if args.color == "random":
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    print(f"Detected and outlined {detection_count} objects")
    
    # Display or save the result
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Result saved to {args.output}")
    else:
        # Resize for display if the image is too large
        max_display_height = 800
        if result_image.shape[0] > max_display_height:
            scale = max_display_height / result_image.shape[0]
            width = int(result_image.shape[1] * scale)
            result_image = cv2.resize(result_image, (width, max_display_height))
        
        cv2.imshow("Object Outlines", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

