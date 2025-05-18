import cv2
import numpy as np

def detect_board_lines(image_path):
    """
    Detects and draws the lines of a chessboard on an image by detecting edges.

    Args:
        image_path (str): The path to the chessboard image.

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

        # Draw the contours on the image
        cv2.drawContours(img, filtered_contours, -1, (0, 0, 255), 3)

        # Save the output image
        output_path = "board_lines.jpg"
        cv2.imwrite(output_path, img)

        return output_path

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    image_path = "screenshot.png"  # Replace with your image path
    output_image_path = detect_board_lines(image_path)

    if output_image_path:
        print(f"Board lines detected and saved to: {output_image_path}")
    else:
        print("Failed to detect board lines.")
