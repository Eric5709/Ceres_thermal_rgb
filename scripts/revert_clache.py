import os
import cv2


def enhance_image_with_clahe(input_image_path, output_image_path, clip_limit=2.0, grid_size=(8, 8)):
    # Load the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    revert_img = cv2.bitwise_not(img)
    # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE to the image
    enhanced_img = clahe.apply(revert_img)

    # Save the enhanced image to the output path
    cv2.imwrite(output_image_path, enhanced_img)

if __name__ == "__main__":
    input_folder  =  "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo/Thermal_original"  # Replace with the path to your input folder
    output_folder =  "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo/Thermal"  # Replace with the path to your output folder

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more image extensions as needed
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            enhance_image_with_clahe(input_image_path, output_image_path)