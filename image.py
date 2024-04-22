import cv2
import numpy as np

# Function to create the grid layout with zooming and cropping
def create_grid_layout(grid_cell, grid_shape, frames, cameras):
    cell_width = grid_cell
    cell_height = grid_cell
    grid_width = cell_width * grid_shape[1]
    grid_height = cell_height * grid_shape[0]

    grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, cam in enumerate(cameras):
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        x = col * cell_width
        y = row * cell_height

        frame = next((f for c, f in frames if c == cam), None)
        if frame is not None:
            # Get the zoom factor for the current camera
            zoom_factor = cam.get('zoom', 1.0)

            # Calculate the aspect ratio of the frame
            frame_height, frame_width = frame.shape[:2]
            frame_aspect_ratio = frame_width / frame_height

            # Calculate the aspect ratio of the cell
            cell_aspect_ratio = cell_width / cell_height

            if frame_aspect_ratio > cell_aspect_ratio:
                # Frame is wider than the cell, crop the sides
                new_height = frame_height
                new_width = int(frame_height * cell_aspect_ratio)
            else:
                # Frame is taller than the cell, crop the top and bottom
                new_width = frame_width
                new_height = int(frame_width / cell_aspect_ratio)

            # Calculate the cropping coordinates
            start_x = int((frame_width - new_width) / 2)
            start_y = int((frame_height - new_height) / 2)
            end_x = start_x + new_width
            end_y = start_y + new_height

            # Crop the frame based on the calculated coordinates
            cropped_frame = frame[start_y:end_y, start_x:end_x]

            # Resize the cropped frame to fit the cell dimensions
            resized_frame = cv2.resize(cropped_frame, (int(cell_width * zoom_factor), int(cell_height * zoom_factor)))

            # Calculate the coordinates to place the resized frame in the center of the cell
            # x_offset = int((cell_width - resized_frame.shape[1]) / 2)
            # y_offset = int((cell_height - resized_frame.shape[0]) / 2)

            # Ensure the resized frame fits within the grid cell
            resized_frame = cv2.resize(resized_frame, (min(resized_frame.shape[1], cell_width), min(resized_frame.shape[0], cell_height)))

            # Place the resized frame in the grid cell
            grid_frame[y:y+resized_frame.shape[0], x:x+resized_frame.shape[1]] = resized_frame

    return grid_frame


def cell_index_from_detected_objects(grid_cell, grid_shape, names, detected_objects):
    # Get the dimensions of each cell in the grid
    cell_width = grid_cell
    cell_height = grid_cell

    # Iterate over the detected bounding boxes
    for *xyxy, conf, cls in reversed(detected_objects):
        # Calculate the center coordinates of the bounding box
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2

        # Determine the cell index based on the center coordinates
        col = int(center_x / cell_width)
        row = int(center_y / cell_height)

        # Calculate the camera index based on the row and column
        cam_index = row * grid_shape[1] + col
        label = names[int(cls)]

        # Get the corresponding camera
        if label == 'person':
            return cam_index