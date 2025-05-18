import cv2
import numpy as np
import argparse
import os

def detect_board_lines(image_path, output_path):
    """
    Detects and draws the lines of a chessboard on an image by detecting edges.

    Args:
        image_path (str): The path to the chessboard image.
        output_path (str): The path to save the output image.

    Returns:
        str: The path to the output image with the detected lines.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not open or read the image at: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 150, 250, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:
                    filtered_contours.append(contour)

        if filtered_contours:
            # Find the contour with the largest area
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            # Get the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the image
            cropped_img = img[y:y+h, x:x+w]

            # Save the cropped image
            cv2.imwrite(output_path, cropped_img)

            return output_path
        else:
            print("No suitable contours found.")
            return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect board lines in images.")
    parser.add_argument("--input_dir", help="Path to the directory containing the images.", default="test")
    parser.add_argument("--output_dir", help="Path to the directory to save the output images.", default="test/output")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(args.input_dir, image_file)
        output_path = os.path.join(args.output_dir, f"{i+1}.jpg")
        output_image_path = detect_board_lines(image_path, output_path)

        if output_image_path:
            print(f"Board lines detected and saved to: {output_image_path}")
        else:
            print(f"Failed to detect board lines for {image_file}.")
