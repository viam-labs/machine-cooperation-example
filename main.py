import math
import os
import asyncio
import asyncio
import cv2
import numpy as np
import io
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.board import Board
from viam.components.motor import Motor
from viam.components.base import Base
from viam.components.encoder import Encoder
from viam.components.camera import Camera
from datetime import datetime

load_dotenv()  # This loads the environment variables from .env

# Define the outputs directory path
outputs_dir = "outputs"

# Check if the outputs directory exists, if not create it
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)


CAMERA_DEVICE_ADDRESS = os.getenv("CAMERA_DEVICE_ADDRESS")
CAMERA_DEVICE_LOC_SECRET = os.getenv("CAMERA_DEVICE_LOC_SECRET")
ROVER_DEVICE_LOC_SECRET = os.getenv("ROVER_DEVICE_LOC_SECRET")
ROVER_DEVICE_ADDRESS = os.getenv("ROVER_DEVICE_ADDRESS")


async def connectToCameraSystem():
    creds = Credentials(
        type='robot-location-secret',
        payload=CAMERA_DEVICE_LOC_SECRET)
    opts = RobotClient.Options(
        refresh_interval=0,
        dial_options=DialOptions(credentials=creds)
    )
    return await RobotClient.at_address(CAMERA_DEVICE_ADDRESS, opts)



async def connectToRover():
    creds = Credentials(
        type='robot-location-secret',
        payload=ROVER_DEVICE_LOC_SECRET)
    opts = RobotClient.Options(
        refresh_interval=0,
        dial_options=DialOptions(credentials=creds)
    )
    return await RobotClient.at_address(ROVER_DEVICE_ADDRESS, opts)

# Function to calculate the centroid of a triangle
def calculate_triangle_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / 3
    centroid_y = sum(y_coords) / 3
    return int(centroid_x), int(centroid_y)

# Function to get the two closest vertices from the triangle vertices
def get_two_closest_vertices(vertices):
    # Calculate pairwise distances between the vertices
    distances = {}
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            dist = math.dist(vertices[i], vertices[j])
            distances[(i, j)] = dist

    # Sort the distances and get the pair with the smallest distance
    closest_pair = min(distances, key=distances.get)
    return vertices[closest_pair[0]], vertices[closest_pair[1]]

# Function to draw the perpendicular line from the midpoint of the closest sides
def draw_perpendicular_from_midpoint(image, pt1, pt2):
    midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    
    # Drawing a perpendicular line of length 100 (can be changed)
    # Using the negative reciprocal of the slope of the line between pt1 and pt2
    end_point = (int(midpoint[0] - 100 * dy), int(midpoint[1] + 100 * dx))
    cv2.line(image, midpoint, end_point, (255, 0, 0), 2)

# Function to draw the angle line
def draw_angle_line(image, top_point, angle_deg, length=100):
    # Convert the angle in degrees to radians
    angle_rad = math.radians(angle_deg)
    
    # Calculate the end point of the line
    end_x = int(top_point[0] + length * math.sin(angle_rad))
    end_y = int(top_point[1] - length * math.cos(angle_rad))  # minus because y-coordinates are inverted in image
    
    # Draw the line on the image
    cv2.line(image, (top_point[0], top_point[1]), (end_x, end_y), (255, 0, 0), 2)
    return image

# Function to calculate the angle of the triangle
def calculate_triangle_angle(point1, point2, point3):
    # Calculate the mid-point of the base of the triangle
    mid_base_x = (point2[0] + point3[0]) / 2
    mid_base_y = (point2[1] + point3[1]) / 2
    
    # Vector from the mid-point of the base to the top point of the triangle
    vector_x = point1[0] - mid_base_x
    vector_y = point1[1] - mid_base_y
    
    # Calculate the angle with respect to the vertical axis
    angle = np.arctan2(vector_y, vector_x)
    angle_deg = np.degrees(angle)
    
    return angle_deg if angle_deg >= 0 else angle_deg + 360

def calculate_triangle_angle(point1, point2, point3):
    # Calculate vectors
    vector_a = [point1[0] - point2[0], point1[1] - point2[1]]
    vector_b = [point1[0] - point3[0], point1[1] - point3[1]]
    
    # Compute the dot product
    dot_product = vector_a[0]*vector_b[0] + vector_a[1]*vector_b[1]
    
    # Compute the magnitudes of the vectors
    magnitude_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
    magnitude_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)
    
    # Compute the angle using the arccosine of the dot product over the magnitudes product
    angle_rad = math.acos(dot_product / (magnitude_a * magnitude_b))
    
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

# Function to process the image and detect the triangle
async def process_image(image_data):
    # Convert the bytes to a numpy array
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    
    # Decode the image from the array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    

    # Generate a directory name based on the current datetime
    # Format as 'YYYYMMDD-HHMMSS'
    dir_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    full_dir_path = os.path.join(outputs_dir, dir_name)  # Change the path to be inside 'outputs'
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)

    # Save the original image for debugging
    cv2.imwrite(os.path.join(full_dir_path, 'original_image.jpg'), image)


    # Save the original image for debugging
    cv2.imwrite(os.path.join(dir_name, 'original_image.jpg'), image)


    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Given HSV values (we convert these from 0-360 & percentages to 0-179 & 0-255 for OpenCV)
    given_hue = 195 / 2  # Since OpenCV's hue max is 179
    given_saturation = 96.3 / 100 * 255
    given_value = 83.9 / 100 * 255

    # Define the range of the specific blue color in HSV
    # You might still need to adjust the sensitivity to get the desired range
    sensitivity = 15
    lower_blue = np.array([given_hue - sensitivity, given_saturation - (sensitivity * 2.55), given_value - (sensitivity * 2.55)])
    upper_blue = np.array([given_hue + sensitivity, given_saturation + (sensitivity * 2.55), given_value + (sensitivity * 2.55)])
    lower_blue = np.clip(lower_blue, 0, 255).astype(np.uint8)
    upper_blue = np.clip(upper_blue, 0, 255).astype(np.uint8)

    # Create a mask that captures areas of the image with the defined blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)

    # Apply threshold to the masked grayscale image
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)



    # Save the threshold image for debugging
    cv2.imwrite(os.path.join(dir_name, 'threshold.jpg'), thresh)


    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw all contours on the image for debugging purposes
    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0,255,0), 2)
    cv2.imwrite(os.path.join(dir_name, 'contours.jpg'), debug_image)

    # Variable to hold the centroid
    triangle_centroid = None


    for cnt in contours:
        print(f"len: {len(cnt)}")  # This line is already in your code
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Draw each individual contour with the number of points found
        contour_image = image.copy()
        cv2.drawContours(contour_image, [cnt], -1, (0,255,0), 2)
        cv2.putText(contour_image, f"Points: {len(approx)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        contour_filename = f'contour_{len(approx)}.jpg'
        cv2.imwrite(os.path.join(dir_name, contour_filename), contour_image)

        if len(approx) == 3:
            points = [tuple(pt[0]) for pt in approx]
            points_sorted_by_y = sorted(points, key=lambda pt: pt[1])
            top_point = points_sorted_by_y[0]
            base_points = points_sorted_by_y[1:]
            
            # Identify the two longest sides of the triangle
            sides = [
                (top_point, base_points[0]),
                (top_point, base_points[1]),
                (base_points[0], base_points[1])
            ]
           

            # Get two closest vertices
            v1, v2 = get_two_closest_vertices(points)
            # Draw the perpendicular line from the midpoint of these vertices
            draw_perpendicular_from_midpoint(image, v1, v2)

             # Get the centroid of the triangle
            triangle_centroid = calculate_triangle_centroid(points)
            
            # Draw a small circle at the centroid for visualization
            cv2.circle(image, triangle_centroid, 5, (0, 0, 255), -1)
            
            # Save the processed image for debugging
            cv2.imwrite(os.path.join(dir_name, 'processed_image.jpg'), image)

    return None, image


async def main():

    #connect to machines
    cameraSystem = await connectToCameraSystem();
    #rover = await connectToRover()

    #print out camera machine
    cam = Camera.from_robot(cameraSystem, "cam")
    raw_image = await cam.get_image()
    

   # Process the image
    angle, processed_image = await process_image(raw_image.data)



    if angle is not None:
        print(f"Triangle is pointing at angle: {angle} degrees")
    else:
        print("No triangle detected.")
  

    # Don't forget to close the robot when you're done!
    await cameraSystem.close()
    #await rover.close()

if __name__ == '__main__':
    asyncio.run(main())

