# 🎨 Contours - Object Detection & Outline Drawing

A sophisticated Python script that detects objects in images and traces their outlines using Mask R-CNN v2. 

## ✨ Features

- 🔍 Object detection using state-of-the-art Mask R-CNN ResNet50 FPN v2
- 🎯 Customizable confidence threshold for detections
- 🌈 Flexible outline color options (random, white, red, green, blue, or custom RGB)
- 📏 Adjustable outline thickness
- 🏷️ Optional class labels and confidence scores display
- 📊 Class filtering capabilities
- 🖼️ Support for various image formats

## 🚀 Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

## 💻 Usage

```bash
python contours.py --image <path_to_image> [options]
```

### 🎮 Command Line Options

- `--image`: Path to the input image (required)
- `--output`: Path to save the output image (optional)
- `--thickness`: Outline thickness (default: 2)
- `--color`: Outline color (default: random)
    - Options: random, white, red, green, blue, or R,G,B values
- `--conf`: Confidence threshold (default: 0.5)
- `--classes`: Filter specific classes (comma-separated)
- `--show-labels`: Display class names and confidence scores

### 📝 Examples

Basic usage:
```bash
python contours.py --image input.jpg
```

Save output with custom settings:
```bash
python contours.py --image input.jpg --output result.jpg --thickness 3 --color red --show-labels
```

Filter specific classes:
```bash
python contours.py --image input.jpg --classes person,car,dog --conf 0.7
```

## 🎯 Supported Classes

Detects 80+ classes from the COCO dataset including:
- 👤 People
- 🚗 Vehicles
- 🐕 Animals
- 🪑 Furniture
- 📱 Electronics
and many more!

## 📄 License

This project is open source and available under the MIT License.

