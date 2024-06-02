import cv2
import numpy as np
from ultralytics import YOLO
import os
import streamlit as st
import pandas as pd

# Function to count vehicles in a frame
def count_vehicles(frame, model, conf_threshold=0.5):
    results = model.predict(frame, conf=conf_threshold, imgsz=320)
    vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0}
    for result in results[0].boxes:
        cls = int(result.cls)
        if cls == 2:  # Car
            vehicle_counts['car'] += 1
        elif cls == 5:  # Bus
            vehicle_counts['bus'] += 1
        elif cls == 7:  # Truck
            vehicle_counts['truck'] += 1
    return vehicle_counts, results

def draw_table_on_frame(frame, data):
    table_img = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
    table_img.fill(255)

    # Write the table headers
    headers = list(data.keys())
    col_width = frame.shape[1] // len(headers)

    for i, header in enumerate(headers):
        cv2.putText(table_img, header, (i * col_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the table data
    for i, (key, values) in enumerate(data.items()):
        for j, value in enumerate(values):
            cv2.putText(table_img, str(value), (i * col_width + 10, (j + 1) * 30 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    combined_img = np.vstack((frame, table_img))
    return combined_img

def main():
    # Title
    st.title("Vehicle Counting and Pollution Estimation App")

    # File uploader
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        # Load a custom trained YOLO model
        model = YOLO(r"C:\Users\shiva\OneDrive\Desktop\CNN\Project2\Problem3\Code\yolov8n.pt")  # Load the custom-trained model

        # Load the video
        video_bytes = uploaded_file.read()
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            return

        # Get the video's frames per second (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Process every second

        # Create a placeholder for the image
        placeholder = st.empty()

        # CO2 emissions values (g/km)
        emissions = {'car': 181.74, 'bus': 215.63, 'truck': 284.30}

        # Read until video is completed
        frame_count = 0
        processed_frame_count = 0
        total_vehicles = {'car': 0, 'bus': 0, 'truck': 0}
        unique_vehicles = {'car': set(), 'bus': set(), 'truck': set()}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every frame_interval frame
            if frame_count % frame_interval == 0:
                # Count vehicles in the current frame
                vehicle_counts, results = count_vehicles(frame, model)
                processed_frame_count += 1

                # Update unique vehicles set with current frame detections
                for result in results[0].boxes:
                    cls = int(result.cls)
                    bbox = tuple(result.xyxy[0].cpu().numpy().astype(int))
                    if cls == 2:
                        unique_vehicles['car'].add(bbox)
                    elif cls == 5:
                        unique_vehicles['bus'].add(bbox)
                    elif cls == 7:
                        unique_vehicles['truck'].add(bbox)

                # Update the total vehicles count
                total_vehicles['car'] += vehicle_counts['car']
                total_vehicles['bus'] += vehicle_counts['bus']
                total_vehicles['truck'] += vehicle_counts['truck']

                # Calculate total pollution
                total_pollution = {
                    'car': total_vehicles['car'] * emissions['car'],
                    'bus': total_vehicles['bus'] * emissions['bus'],
                    'truck': total_vehicles['truck'] * emissions['truck']
                }

                # Prepare data for the table
                data = {
                    'Vehicle Type': ['Cars', 'Buses', 'Trucks'],
                    'Total Vehicles': [total_vehicles['car'], total_vehicles['bus'], total_vehicles['truck']],
                    'CO2g/km': [emissions['car'], emissions['bus'], emissions['truck']],
                    'Total Pollution (g)': [total_pollution['car'], total_pollution['bus'], total_pollution['truck']]
                }

                # Draw bounding boxes and labels on the frame
                for result in results[0].boxes:
                    bbox = result.xyxy[0].cpu().numpy().astype(int)
                    cls = int(result.cls)
                    label = model.names[cls]
                    if label in ['car', 'bus', 'truck']:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Draw table on the frame
                frame_with_table = draw_table_on_frame(frame, data)

                # Display the frame with labels
                placeholder.image(frame_with_table, channels="BGR")

            frame_count += 1

        # When everything done, release the video capture object
        cap.release()

        # Print total unique vehicles detected in the video
        st.write(f"Total unique vehicles detected in the video: Cars - {len(unique_vehicles['car'])}, Buses - {len(unique_vehicles['bus'])}, Trucks - {len(unique_vehicles['truck'])}")

        # Delete temporary video file
        os.remove(video_path)

if __name__ == "__main__":
    main()
