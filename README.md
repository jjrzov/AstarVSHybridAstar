## Overview
This repo contains a demo script for comparing A* and Hybrid A* within the CARLA autonomous driving simulator. Implementations for A*, Hybrid A*, Reeds-Shepp curves, and the Stanley Controller are done by zhm-real and included in their repo:

Basic Path Planning Algorithms: (https://github.com/zhm-real/MotionPlanning) MotionPlanning

To setup the demo download these dependencies:
    pip install carla numpy scipy heapdict

To avoid path dependency issues keep folder structure as shown in this repo.

## Running the Demo
Step 1:
Download CARLA from CARLA website and Run CarlaUE4.exe. Then wait for CARLA window to fully load

Step 2:
Run the demo script with
    python .\parking_lot.py

## Demo Information
The script will automatically connect to CARLA and load Town05. Spectator perspective will spawn to the right of the parking lot, adjust view to see parking lot.

The script will setup A* and Hybrid A* based off the parking lot and run the planners with visualizers. The red blocks are the A* path, the green are the Hybrid A* path, and the grey boxes are obstacles. 

Once the planning has been done for both planners, the terminal will ask you to press ENTER to set the car to move. The first press of ENTER will run Hybrid A* and once completed respawn the vehcile at the start position. Once the terminal asks you again to press ENTER, and ENTER is pressed the A* path will be followed by the car. 

Upon completion qualtitaive metrics will be printed out in the terminal for you to compare the perfomances.