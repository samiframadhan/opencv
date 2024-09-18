import cv2
import os

# Function to extract frames from video and save them as images
def video_to_frames(video_path, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    success, frame = video_capture.read()

    while success:
        # Save the frame as a JPEG file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Print progress
        print(f"Saved {frame_filename}")

        # Read the next frame
        success, frame = video_capture.read()
        frame_count += 1
        if frame_count == 20:
            break

    # Release the video capture object
    video_capture.release()
    print(f"Extraction complete. {frame_count} frames extracted.")

# Example usage
video_path = "VideoTrack.mp4"  # Replace with the actual video file path
output_folder = "data"     # Replace with your desired output folder

video_to_frames(video_path, output_folder)
