import os
import cv2
import pandas as pd

diter_rgb = "/home/jsh/robot_ws/livox_preprocessed/rosbag2_2023_03_09-13_42_46.png"  # Replace with the path to your output folder
diter_thermal = "/home/jsh/robot_ws/livox_preprocessed/00000.png"  # Replace with the path to your input folder

csv_file = "/home/jsh/Koide.csv"
data = pd.read_csv(csv_file)

thermal_image = cv2.imread(diter_thermal, cv2.IMREAD_GRAYSCALE)
RGB_image     = cv2.imread(diter_rgb)

# thermal_image_resized = cv2.resize(thermal_image, (RGB_image.shape[1], RGB_image.shape[0]))

for _, row in data.iterrows():
    rgb_x = int(row['rgb_x'])
    rgb_y = int(row['rgb_y'])
    thermal_x = int(row['thermal_x'])
    thermal_y = int(row['thermal_y'])
    projected_x = int(row['projected_x'])
    projected_y = int(row['projected_y'])
    # print("*************************")
    # print(thermal_x ,thermal_y, projected_x, projected_y)
    cv2.circle(thermal_image, (thermal_x, thermal_y), 5, (0, 0, 255), -1)  
    cv2.circle(RGB_image, (rgb_x, rgb_y), 5, (0, 0, 255), -1)  
    cv2.circle(RGB_image, (projected_x, projected_y), 5, (0, 255, 0), -1)  

imS = cv2.resize(RGB_image, (1080, 1080))                # Resize image
rotated_image = cv2.rotate(imS, cv2.ROTATE_90_CLOCKWISE)

cv2.namedWindow("RGB")
cv2.imshow("RGB", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()