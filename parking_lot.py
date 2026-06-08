'''
A* vs Hybrid A* Comparison Script

This script compares A* and Hybrid A* in a parking lot using the CARLA simulator.
The implementations for A*, Hybrid A*, Reeds-Shepp Curves, and the Stanley Controller
are all from an open source repo by zhm-real. 

To compare the planners, choose one of the trial runs (1-5) and see the paths
visualized in CARLA. Red blocks are A* path, Green blocks are Hybrid A*, and the
grey blocks are obstacles. Further details can be found in the report attached in 
the GitHub repo for this project.
'''

import sys
import os
import math
import time
import numpy as np
import carla

# ── zhm-real paths ────────────────────────────────────────────────────────────
sys.path.insert(0, './MotionPlanning/HybridAstarPlanner')
sys.path.insert(0, './MotionPlanning/Control')

import hybrid_astar, astar
from Stanley import Node, Trajectory, front_wheel_feedback_control, pid_control, C

# Config settings
CARLA_HOST   = "localhost"
CARLA_PORT   = 2000

# Parking lot bounds (from get_environment_objects output)
LOT_MIN_X, LOT_MAX_X = -39.0, 19.0
LOT_MIN_Y, LOT_MAX_Y = -50.0, -10.0

# Demo trials for comparing the planners
TRIALS = {
    1: ((-33.0, -35.0, 90.0),  ( 4.0,  -32.0,   0.0)),  # Obstacles
    2: ((-31.0, -13.0,  0.0),  (11.0,  -13.0,   0.0)),  # No obstacles
    3: ((-21.0, -15.0,-90.0),  (16.0,  -41.0, -90.0)),  # More obstacles
    4: ((-31.0, -13.0,  0.0),  (15.0,  -41.0, -90.0)),  # Diagonal
    5: ((-35.0, -44.0, 90.0),  (-25.0, -26.0,   0.0)),  # Tight Left
}

ACTIVE_TRIAL = 1    # Select trial to run
START, GOAL = TRIALS[ACTIVE_TRIAL]

# Planner config
RESOLUTION     = 1.0    # Discretized the parking lot to 1m/cell
YAW_RESOLUTION = np.deg2rad(15.0)
ROBOT_RADIUS   = 1.0          # obstacle inflation for A*
OBSTACLE_PADDING = 0.0        # extra padding around each car bounding box [m]
BORDER_PAD     = 0.5          # border wall inset from lot edge [m]

TARGET_SPEED   = 3.0          # m/s — slow for parking lot
FOLLOW_HYBRID  = True         # drive the Hybrid A* path
FOLLOW_ASTAR   = True         # attempt to drive the A* path (will likely fail)
LIFE_TIME      = 120.0        # debug drawing lifetime [s]



def setup_carla():
    # Connect to CARLA and spawn vehicle
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town05') # Town 5 has the viable parking lot
    carla_map = world.get_map()

    # Get vehicle GPS
    bps = world.get_blueprint_library().filter('vehicle.*')
    vehicle_bp = bps.find('vehicle.audi.etron')

    # Spawn main vehicle at inside the parking lot
    spawn_point = carla.Transform(
        carla.Location(x=START[0], y=START[1], z=0.5),
        carla.Rotation(yaw=START[2])
    )
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    return client, world, carla_map, vehicle


def addObstacle(seen, obstacle_x, obstacle_y, point):
    # Add obstacle x and y at given point to respective obstacle list
    key = (round(point[0] / RESOLUTION), round(point[1] / RESOLUTION))
    
    if key not in seen:
        # Only want to add obstacles not seen yet
        seen.add(key)
        obstacle_x.append(point[0])
        obstacle_y.append(point[1])


def getStaticCars(world):
    # Return a list of all the car objects within the parking lot
    all_cars = []
    parking_lot_cars = []

    # Parking lot has Cars and Trucks
    for label in [carla.CityObjectLabel.Car, carla.CityObjectLabel.Truck]:
        all_cars += world.get_environment_objects(label)

    # Get bounding box location of all cars
    for car in all_cars:
        car_x, car_y = car.bounding_box.location.x, car.bounding_box.location.y

        if LOT_MIN_X < car_x < LOT_MAX_X and LOT_MIN_Y < car_y < LOT_MAX_Y:
            # Only want cars inside parking lot
            parking_lot_cars.append(car)

    return parking_lot_cars


def createObstacleList(world):
    # Create list of all obstacles for zhm-real's implementation
    obstacle_x, obstacle_y = [], []
    seen = set()

    # Get car objects in parking lot
    parked_cars = getStaticCars(world)
    print(f"[Obstacles] Found {len(parked_cars)} static cars in parking lot")

    # Place obstacle points around each parked car
    for car in parked_cars:
        loc, ext = car.bounding_box.location, car.bounding_box.extent
        
        ext_x = abs(ext.x) + OBSTACLE_PADDING
        ext_y = abs(ext.y) + OBSTACLE_PADDING

        # Sample a grid of points covering the bounding box
        for dx in np.arange(-ext_x, ext_x + RESOLUTION, RESOLUTION):
            for dy in np.arange(-ext_y, ext_y + RESOLUTION, RESOLUTION):
                point = (loc.x + dx, loc.y + dy)
                addObstacle(seen, obstacle_x, obstacle_y, point)

    # Place obstacle points around parking lot
    min_x = LOT_MIN_X + BORDER_PAD
    max_x = LOT_MAX_X - BORDER_PAD
    min_y = LOT_MIN_Y + BORDER_PAD
    max_y = LOT_MAX_Y - BORDER_PAD

    # Interpolate x and y points for obstacle points to create a perimeter
    for x in np.arange(min_x, max_x, RESOLUTION):
        addObstacle(seen, obstacle_x, obstacle_y, (x, min_y))
        addObstacle(seen, obstacle_x, obstacle_y, (x, max_y))

    for y in np.arange(min_y, max_y, RESOLUTION):
        addObstacle(seen, obstacle_x, obstacle_y, (min_x, y))
        addObstacle(seen, obstacle_x, obstacle_y, (max_x, y))

    print(f"[Obstacles] Total: {len(obstacle_x)} points "
          f"({len(parked_cars)} cars + border wall)")
    
    return obstacle_x, obstacle_y


def run_hybrid_astar(obstacles_x, obstacles_y):
    # Run zhm-real's implementation of Hybrid A*

    # Get start and goal pose
    sx, sy, syaw = START[0], START[1], math.radians(START[2])
    gx, gy, gyaw = GOAL[0],  GOAL[1],  math.radians(GOAL[2])

    print("\n[Hybrid A*] Planning...")

    t0   = time.time()  # Record time to plan
    path = hybrid_astar.hybrid_astar_planning(
        sx, sy, syaw,
        gx, gy, gyaw,
        obstacles_x, obstacles_y,
        RESOLUTION, YAW_RESOLUTION
    )
    elapsed = time.time() - t0

    if path is None:
        print(f"[Hybrid A*] No path found ({elapsed:.2f}s)")
        return None, elapsed

    print(f"[Hybrid A*] Found in {elapsed:.2f}s — {len(path.x)} waypoints")
    return path, elapsed


def run_astar(obstacles_x, obstacles_y):
    # Run zhm-real's A* implementation

    # Get start and goal pose (A* doesn't need heading)
    sx, sy = START[0], START[1]
    gx, gy = GOAL[0],  GOAL[1]

    print("\n[A*] Planning...")

    t0 = time.time()    # Begin timer for planning time
    pathx, pathy = astar.astar_planning(
        sx, sy, 
        gx, gy,
        obstacles_x, obstacles_y,
        RESOLUTION, 
        ROBOT_RADIUS
    )
    elapsed = time.time() - t0

    if not pathx:
        print(f"[A*] No path found ({elapsed:.2f}s)")
        return None, None, elapsed

    print(f"[A*] Found in {elapsed:.2f}s — {len(pathx)} waypoints")
    return pathx, pathy, elapsed


def visualise(world, obstacles_x, oy, hybrid_path, astar_x, astar_y):
    # Visualize the paths from each planner

    # Obstacles in white
    for x, y in zip(obstacles_x, oy):
        world.debug.draw_point(carla.Location(x=x, y=y, z=0.3),
                               size=0.06,
                               color=carla.Color(200, 200, 200),
                               life_time=LIFE_TIME)
    
    # Start in orange
    world.debug.draw_point(carla.Location(x=START[0], y=START[1], z=1.0),
                           size=0.2,
                           color=carla.Color(255, 140, 0),
                           life_time=LIFE_TIME)
    
    # Goal in blue
    world.debug.draw_point(carla.Location(x=GOAL[0], y=GOAL[1], z=1.0),
                           size=0.2,
                           color=carla.Color(0, 0, 255),
                           life_time=LIFE_TIME)
    
    # Hybrid A* in green
    if hybrid_path:
        for x, y in zip(hybrid_path.x, hybrid_path.y):
            world.debug.draw_point(carla.Location(x=x, y=y, z=0.5),
                                   size=0.1,
                                   color=carla.Color(0, 220, 0),
                                   life_time=LIFE_TIME)
        print("[Viz] Hybrid A* path drawn in GREEN")

    # A* in red
    if astar_x:
        for x, y in zip(astar_x, astar_y):
            world.debug.draw_point(carla.Location(x=x, y=y, z=0.5),
                                   size=0.1,
                                   color=carla.Color(220, 0, 0),
                                   life_time=LIFE_TIME)
        print("[Viz] A* path drawn in RED")


def path_length(xs, ys):
    # Calculate total path length
    return sum(math.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i])
               for i in range(len(xs)-1))

def path_smoothness(xs, ys):
    # Calculate average and max turning angle between consecutive segments in the path
    angles = []

    for i in range(1, len(xs) - 1):
        # Vector from previous to current
        v1x = xs[i]   - xs[i-1]
        v1y = ys[i]   - ys[i-1]
    
        # Vector from current to next
        v2x = xs[i+1] - xs[i]
        v2y = ys[i+1] - ys[i]

        # Angle between vectors
        cos_a = (v1x*v2x + v1y*v2y) / (
            math.hypot(v1x, v1y) * math.hypot(v2x, v2y) + 1e-6)
        
        cos_a = max(-1.0, min(1.0, cos_a))  # clamp incase
        angles.append(math.acos(cos_a))

    if not angles:
        return 0.0, 0.0
    
    return float(np.mean(angles)), float(np.max(angles))


def print_comparison(hybrid_path, hybrid_time, astar_x, astar_y, astar_time):
    # Print qualtitiave metrics

    print("\n" + "="*58)
    print(f"{'METRIC':<32} {'HYBRID A*':>12} {'A*':>12}")
    print("="*58)

    h_len = path_length(hybrid_path.x, hybrid_path.y) if hybrid_path else 0
    a_len = path_length(astar_x, astar_y) if astar_x else 0
    h_avg, h_max = path_smoothness(hybrid_path.x, hybrid_path.y)
    a_avg, a_max = path_smoothness(astar_x, astar_y)

    print(f"{'Path length (m)':<32} {h_len:>12.1f} {a_len:>12.1f}")
    print(f"{'Planning time (s)':<32} {hybrid_time:>12.2f} {astar_time:>12.2f}")
    print(f"{'Avg turning angle (rad)':<32} {h_avg:>12.4f} {a_avg:>12.4f}")
    print(f"{'Max turning angle (rad)':<32} {h_max:>12.4f} {a_max:>12.4f}")
    print(f"{'Waypoints':<32} "
          f"{len(hybrid_path.x) if hybrid_path else 0:>12} "
          f"{len(astar_x) if astar_x else 0:>12}")
    print("="*58)


def follow_path(world, vehicle, path_x, path_y, path_yaw, label):
    # Follow path from the planners and feed into zhm-real's implementation of Stanley controller

    # No yaw if A*, thus approximate from path direction
    if path_yaw is None:
        yaws = []

        for i in range(len(path_x)):
            if i < len(path_x) - 1:
                yaws.append(math.atan2(path_y[i+1] - path_y[i],
                                       path_x[i+1] - path_x[i]))
            else:
                yaws.append(yaws[-1])
        
        path_yaw = yaws

    ref_path = Trajectory(path_x, path_y, path_yaw)

    tf    = vehicle.get_transform()
    vel   = vehicle.get_velocity()
    node  = Node(x   = tf.location.x,
                 y   = tf.location.y,
                 yaw = math.radians(tf.rotation.yaw),
                 v   = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

    gx, gy = GOAL[0], GOAL[1]
    print(f"[Control] Following {label} path...")
    world.wait_for_tick()

    while True:
        world.wait_for_tick()

        tf    = vehicle.get_transform()
        vel   = vehicle.get_velocity()
        node.x   = tf.location.x
        node.y   = tf.location.y
        node.yaw = math.radians(tf.rotation.yaw)
        node.v   = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        dist = math.hypot(node.x - gx, node.y - gy)
        if dist < 2.0:
            print(f"[Control] {label} — Goal reached!")
            break

        delta, _ = front_wheel_feedback_control(node, ref_path)
        ai       = pid_control(TARGET_SPEED, node.v, dist)
        throttle = max(0.0, float(ai))
        brake    = max(0.0, float(-ai)) * 0.3
        steer    = float(np.clip(delta / C.MAX_STEER, -1.0, 1.0))

        vehicle.apply_control(carla.VehicleControl(
            throttle = float(np.clip(throttle, 0.0, 1.0)),
            steer    = steer,
            brake    = float(np.clip(brake,    0.0, 1.0)),
        ))

    vehicle.apply_control(carla.VehicleControl(brake=1.0))
    time.sleep(2.0)

    # Reset vehicle to start for next run
    vehicle.set_transform(carla.Transform(
        carla.Location(x=START[0], y=START[1], z=0.5),
        carla.Rotation(yaw=START[2])
    ))
    world.wait_for_tick()


def main():
    client, world, carla_map, vehicle = setup_carla()

    try:
        # Build obstacle list from static lot cars and border wall
        ox, oy = createObstacleList(world)

        # Run both planners on identical inputs
        hybrid_path, hybrid_time = run_hybrid_astar(ox, oy)
        astar_x, astar_y, astar_time = run_astar(ox, oy)

        # Draw everything in CARLA
        visualise(world, ox, oy, hybrid_path, astar_x, astar_y)

        # Print comparison table
        if hybrid_path or astar_x:
            print_comparison(hybrid_path, hybrid_time,
                             astar_x, astar_y, astar_time)

        # Follow Hybrid A* path
        if FOLLOW_HYBRID and hybrid_path:
            input("\nPress Enter to drive Hybrid A* path (GREEN)...")
            follow_path(world, vehicle,
                        hybrid_path.x, hybrid_path.y, hybrid_path.yaw,
                        "Hybrid A*")

        # Follow A* path
        if FOLLOW_ASTAR and astar_x:
            input("\nPress Enter to attempt A* path (RED)...")
            follow_path(world, vehicle,
                        astar_x, astar_y, None,
                        "A*")

    finally:
        # Cleanup
        print("[CARLA] Cleaning up...")
        vehicle.destroy()
        print("[CARLA] Done.")


if __name__ == "__main__":
    main()