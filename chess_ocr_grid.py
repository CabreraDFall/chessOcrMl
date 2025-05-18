import cv2
import numpy as np
import argparse
import os

def draw_grid(image_path, output_path, grid_size=8):
    """
    Draws an 8x8 grid on an image.

    Args:
        image_path (str): The path to the image.
        output_path (str): The path to save the output image.
        grid_size (int): The number of rows and columns in the grid.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not open or read the image at: {image_path}")

        height, width = img.shape[:2]

        # Calculate cell size
        cell_width = width // grid_size
        cell_height = height // grid_size

        # Draw vertical lines
        for i in range(1, grid_size):
            x = i * cell_width
            cv2.line(img, (x, 0), (x, height), (0, 0, 255), 2)

        # Draw horizontal lines
        for i in range(1, grid_size):
            y = i * cell_height
            cv2.line(img, (0, y), (width, y), (0, 0, 255), 2)

        # Save the image
        cv2.imwrite(output_path, img)
        return output_path

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

    image_file = "output/1.jpg"
    image_path = os.path.join(args.input_dir, image_file)
    output_path = os.path.join(args.output_dir, "grid.jpg")
    output_image_path = draw_grid(image_path, output_path)

    if output_image_path:
        print(f"Grid drawn and saved to: {output_image_path}")
    else:
        print(f"Failed to draw grid for {image_file}.")
