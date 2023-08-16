import os
import cv2
import pandas as pd

diter_thermal     = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/Diter/Thermal/000000.png"  # Replace with the path to your output folder
diter_rgb         = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/Diter/RGB/000000.png"  # Replace with the path to your input folder

sthereo_thermal   = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo/Thermal/000001.png"
sthereo_rgb_left  = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo/RGB_left/000001.png"
sthereo_rgb_right = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo/RGB_right/000000.png"

csv_file = "/home/jsh/Diter.csv"
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


cv2.namedWindow("Thermal")
cv2.imshow("Thermal", thermal_image)
cv2.namedWindow("RGB")
cv2.imshow("RGB", RGB_image)
cv2.waitKey(0)
cv2.destroyAllWindows()