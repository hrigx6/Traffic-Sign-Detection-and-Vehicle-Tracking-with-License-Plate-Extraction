# Traffic-Sign-Detection-and-Vehicle-Tracking-with-License-Plate-Extraction


This project presents a novel framework for enhancing the safety of autonomous vehicles by leveraging state-of-the-art computer vision techniques. The proposed system combines YOLO v8 object detection, ByteTrack-based vehicle tracking, and OCR-powered license plate extraction to provide a comprehensive solution for traffic sign recognition, vehicle monitoring, and identification.

## Features

- **Traffic Sign Detection**: Robust detection of traffic signs using a fine-tuned YOLOv8 model, achieving high accuracy even in challenging scenarios with varying environmental conditions and diverse sign designs.
- **Vehicle Tracking**: Reliable multi-object tracking of vehicles using the ByteTrack algorithm, enabling consistent tracking across frames and handling occlusions and complex interactions.
- **License Plate Extraction**: Integration of Optical Character Recognition (OCR) for accurate license plate number extraction, facilitating vehicle identification and traffic monitoring applications.
- **Real-time Performance**: Optimized for real-time inference, enabling seamless integration into autonomous vehicle systems and advanced driver assistance systems (ADAS).

## Methodology

1. **Dataset Preparation**: A diverse dataset containing vehicles, traffic signs, and license plates was curated and annotated using Roboflow. Data augmentation techniques were applied to increase diversity.

2. **YOLO-based Traffic Sign Detection**: A pre-trained YOLOv8 model was fine-tuned on the custom dataset, optimizing its performance for traffic sign detection tasks.

3. **Vehicle Tracking with ByteTrack**: The ByteTrack algorithm was integrated with the YOLOv8 detections to achieve robust and consistent vehicle tracking across frames, even in the presence of occlusions and partial visibility.

4. **OCR-based License Plate Detection**: A two-stage approach combining YOLOv8 for license plate localization and EasyOCR for optical character recognition was implemented to accurately extract license plate numbers from vehicle images.

5. **Integrated System**: The traffic sign detection, vehicle tracking, and license plate extraction modules were integrated into a unified framework, enabling comprehensive analysis of traffic scenes for autonomous vehicle applications.

## Results

- **Traffic Sign Detection**: Achieved an average precision of 93.8% for traffic sign detection, outperforming the pre-trained model by a significant margin.
- **Vehicle Tracking**: The ByteTrack algorithm achieved a Multiple Object Tracking Accuracy (MOTA) of 89.4% on the test set, demonstrating its effectiveness in maintaining consistent vehicle identities across frames.
- **License Plate Detection**: The OCR-based approach achieved an average precision of 92.7% for license plate detection, surpassing the performance of the pre-trained YOLOv8 model.

## Conclusion and Future Work

This project has successfully demonstrated promising results in traffic sign detection, vehicle tracking, and license plate extraction, addressing critical challenges in autonomous vehicle safety. However, further improvements are planned to enhance the system's robustness and performance in real-world deployments. Future work includes exploring advanced techniques such as multi-view fusion, occlusion-aware feature extraction, and context-aware reasoning to create a more resilient and error-tolerant system.

## Acknowledgments

We would like to express our sincere gratitude to our advisor, Professor Bruce Maxwell, for his invaluable guidance and support throughout this project.

## Demo Videos

Step-by-step demo videos are available at: https://northeastern-my.sharepoint.com/:f:/g/personal/suryawanshi_h_northeastern_edu/ElrfQ62WgidGmE5netuRAnUBT6RfDdh2PHip2er0TlpFaA?e=oQhvrY

## Code

The main code and the best model weights are included in the repository.
