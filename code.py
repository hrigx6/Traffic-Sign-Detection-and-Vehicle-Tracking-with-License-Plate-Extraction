import os
import random
import pandas as pd
from PIL import Image
import cv2
import supervision as sv
print("supervision.__version__:", sv.__version__)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

import warnings
warnings.filterwarnings('ignore')

import pandas

def get_video_frames_generator(video_path):
    """
    A generator function to yield frames from a video file.

    Args:
    - video_path (str): Path to the video file.

    Yields:
    - frame (numpy.ndarray): Image frame from the video.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

# Directory containing images
Image_dir = 'C:/Users/Hrigved/Downloads/archive/train/images'

# Number of random samples to display
num_samples = 5

# List of image files in the directory
image_files = os.listdir(Image_dir)

print("Number of images:", len(image_files))

# Randomly select images for display
if num_samples > len(image_files) or num_samples < 0:
    print("Error: Invalid number of samples.")
else:
    rand_images = random.sample(image_files, num_samples)

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))  # Modified to plot in a single row

    for i in range(num_samples):
        image = rand_images[i]
        ax = axes[i]
        ax.imshow(plt.imread(os.path.join(Image_dir, image)))
        ax.set_title(f'Image {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Load YOLOv8n model
model = YOLO("yolov8n.pt")
Final_model = YOLO('yolov8n.yaml')._load('best.pt')

# Path to the directory containing the video frames
SOURCE_VIDEO_PATH = f"C:/Users/Hrigved/Downloads/vehicle_count.mp4"

# Create a generator for video frames
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# Iterate over frames (example usage)
for frame in generator:
    # Process each frame here
    pass

# Dictionary mapping class_id to class_name
CLASS_NAMES_DICT = Final_model.model.names

# Class IDs of interest (car, motorcycle, bus, and truck)
selected_classes = [2, 3, 5, 7]

# Create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# Create an instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)

# Acquire the first video frame
iterator = iter(generator)
frame = next(iterator)

# Model prediction on a single frame and conversion to supervision Detections
results = Final_model.predict([frame], verbose=False)[0]

# Convert to Detections
detections = sv.Detections.from_ultralytics(results)

# Only consider class id from selected_classes defined above
detections = detections[np.isin(detections.class_id, selected_classes)]

# Format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for confidence, class_id in zip(detections.confidence, detections.class_id)
]

# Annotate and display frame
annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

# Display the annotated frame
sv.plot_image(annotated_frame, (13, 13))

# Settings for line zone
LINE_START = sv.Point(50, 1500)
LINE_END = sv.Point(3840-50, 1500)

# Target video path
TARGET_VIDEO_PATH = f"C:/Users/Hrigved/Downloads/vehicle-counting-result-with-counter.mp4"

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Create LineZone instance
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=80)

# Create LineZoneAnnotator instance
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=1)

# Define callback function for video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Callback function to process video frames.

    Args:
    - frame (numpy.ndarray): Input frame.
    - index (int): Frame index.

    Returns:
    - annotated_frame (numpy.ndarray): Annotated frame.
    """
    # Model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Only consider class id from selected_classes
    detections = detections[np.isin(detections.class_id, selected_classes)]

    # Tracking detections
    detections = byte_tracker.update_with_detections(detections)

    # Format labels
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    # Annotate frame with traces and bounding boxes
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Update line counter
    line_zone.trigger(detections)

    # Annotate frame with line zone
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# Process the whole video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
