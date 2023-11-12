# Robot System Script Usage Guide

## Overview
This script connects to a camera system and rover system and processes images from the camera to control the behavior of the rover.

## Setup
1. **Install Dependencies**: Ensure Python 3.x is installed along with `opencv-python`, `numpy`, and `python-dotenv`.
2. **Environment Variables**: Create a `.env` file in the script's directory with the following variables:
   - `CAMERA_DEVICE_ADDRESS`: Address of the camera system.
   - `CAMERA_DEVICE_LOC_SECRET`: Secret key for the camera system.
   - `ROVER_DEVICE_LOC_SECRET`: Secret key for the rover system.
   - `ROVER_DEVICE_ADDRESS`: Address of the rover system.

## Running the Script
Execute the script with Python:
```bash
python main.py
```
