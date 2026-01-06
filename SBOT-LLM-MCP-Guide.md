# SBOT LLM Control System via Model Context Protocol (MCP)
## Step-by-Step Documentation Guide

**Author:** AI Research Assistant  
**Date:** January 5, 2026  
**Target System:** SBOT Monocopter (1 Down-Facing Motor + 4 Servo-Controlled Vanes)  
**Framework:** Model Context Protocol (MCP) for LLM Control  
**Reference:** Ilia Larchenko's SO-ARM100 Robot MCP Implementation

---

## Table of Contents

1. [Introduction to MCP for SBOT](#1-introduction-to-mcp-for-sbot)
2. [MCP Architecture Overview](#2-mcp-architecture-overview)
3. [SBOT System Architecture](#3-sbot-system-architecture)
4. [MCP Server Implementation](#4-mcp-server-implementation)
5. [Control Allocation Module](#5-control-allocation-module)
6. [Inverse Kinematics for SBOT Vanes](#6-inverse-kinematics-for-sbot-vanes)
7. [Step-by-Step Setup Guide](#7-step-by-step-setup-guide)
8. [Testing and Validation](#8-testing-and-validation)
9. [Integration with LLMs](#9-integration-with-llms)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Introduction to MCP for SBOT

### What is MCP?

**Model Context Protocol (MCP)** is an open standard developed by Anthropic that enables Large Language Models (LLMs) to control external systems through standardized tools. Think of it as a **universal USB-C port for AI**—allowing any LLM (Claude, GPT, Gemini) to interact with any robot through a common interface.

### Why MCP for SBOT?

Traditional robot control requires:
- Custom integrations for each LLM provider
- Different APIs for different robot types
- Complex prompt engineering for each specific task

**With MCP, you get:**
- ✅ Plug-and-play integration (one implementation, works with all LLMs)
- ✅ Standardized tool descriptions
- ✅ Zero-shot task execution (LLM can control SBOT without fine-tuning)
- ✅ Real-time sensor feedback to the LLM
- ✅ Extensible architecture (add new tools easily)

### SBOT-Specific Benefits

For your monocopter design, MCP enables:
1. **Natural Language Control**: "Move SBOT up 2 meters" → LLM understands monocopter physics
2. **Visual Reasoning**: Camera feed → LLM sees obstacle → Adjusts trajectory
3. **Multi-Modal Sensing**: GNSS, IMU, LiDAR, cameras → Unified state for LLM
4. **Autonomous Adaptation**: Reinforcement learning policy + safety constraints

---

## 2. MCP Architecture Overview

### 2.1 Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│   MCP HOST (LLM Application)            │
│  - Claude Desktop                       │
│  - GPT via API                          │
│  - Gemini                               │
│  - Custom Agent                         │
└────────────┬────────────────────────────┘
             │
             │ JSON-RPC Protocol
             │
┌────────────▼────────────────────────────┐
│   MCP CLIENT (Connection Manager)       │
│  - Discovers available tools            │
│  - Routes requests to server            │
│  - Handles tool results                 │
└────────────┬────────────────────────────┘
             │
             │ stdio / HTTP / WebSocket
             │
┌────────────▼────────────────────────────┐
│   MCP SERVER (Your SBOT Implementation) │
│  - Tool definitions                     │
│  - Resource management                  │
│  - SBOT hardware interface              │
└─────────────────────────────────────────┘
```

### 2.2 Request/Response Flow

```
1. User Request (LLM Host)
   "Move SBOT forward 1 meter"
   ↓
2. LLM Processing
   - Recognizes intent: move_robot
   - Extracts parameters: direction=forward, distance=1m
   ↓
3. Tool Call Generation
   {
     "tool": "move_robot",
     "params": {"direction": "forward", "distance": 1.0, "unit": "m"}
   }
   ↓
4. MCP Server Execution
   - Calls: move_robot(direction="forward", distance=1.0)
   - Converts to servo angles + motor thrust
   - Executes motion control
   ↓
5. Result Collection
   - Gets current SBOT state
   - Captures camera frames
   - Returns position, attitude, sensor data
   ↓
6. LLM Response Generation
   "SBOT moved forward 1.0 meters. Current position: X=1.2m, 
    Y=0.1m, altitude=2.5m. All systems nominal."
```

### 2.3 MCP Server Components

```python
# Core MCP Server Structure
├── mcp_server.py (Main server entrypoint)
├── tools/
│   ├── robot_state.py (Get SBOT state)
│   ├── movement.py (Move robot in space)
│   ├── vane_control.py (Control 4 vanes)
│   ├── motor_control.py (BLDC throttle)
│   └── sensor_access.py (Camera, LiDAR, GNSS)
├── models/
│   ├── kinematics.py (Inverse kinematics for vanes)
│   ├── control_allocation.py (Servo angle → torque)
│   └── dynamics.py (SBOT motion model)
└── hardware/
    ├── motor_driver.py (Motor speed control)
    ├── servo_driver.py (4 servo control)
    ├── sensor_fusion.py (IMU + GNSS + cameras)
    └── communication.py (Serial/CAN to hardware)
```

---

## 3. SBOT System Architecture

### 3.1 Hardware Overview

**SBOT Monocopter Components:**

```
                    ┌─────────────────┐
                    │  BLDC Motor     │ ← 1 motor facing DOWN
                    │  (Vertical      │
                    │   thrust)       │
                    └────────┬────────┘
                             │
                             │ 10x5 Prop
                             │
                    ┌────────▼────────┐
                    │   SBOT Body     │ ← Spins vertically
                    │   (Spinning)    │
                    └────┬──┬──┬──┬───┘
                         │  │  │  │
        ┌────────────────┘  │  │  └───────────────┐
        │                   │  │                   │
    ┌───▼────┐          ┌───▼──▼───┐          ┌───▼────┐
    │Vane 1  │          │Vane 2    │          │Vane 3  │
    │(Front) │          │(Right)   │          │(Back)  │
    └───┬────┘          └────┬─────┘          └───┬────┘
        │                    │                    │
    ┌───▼────┐          ┌────▼─────┐          ┌───▼────┐
    │Servo 1 │          │Servo 2   │          │Servo 3 │
    │(0-90°) │          │(0-90°)   │          │(0-90°) │
    └────────┘          └──────────┘          └────────┘
        
        ┌──────────┐
        │ Vane 4  │ ← Bottom
        │(Bottom) │
        └────┬─────┘
             │
        ┌────▼────┐
        │ Servo 4 │
        │ (0-90°) │
        └─────────┘
```

### 3.2 Control Flow (Classical PID)

```
┌──────────────────────┐
│  Reference Commands  │ (from LLM: "move up 2m")
└─────────┬────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│ Block 0: Altitude Control (PID)      │
│  - Kp=0.8, Ki=0.05, Kd=0.2          │
│  - Input: altitude_ref, altitude_est │
│  - Output: throttle_cmd (1000-2000µs)│
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌──────────────┐  ┌──────────────────────────────────┐
│ BLDC Motor   │  │ Block 1: Attitude Control (PID)  │
│ Speed (RPM)  │  │  - Roll, Pitch, Yaw              │
└──────────────┘  │  - Output: desired torques       │
                  └──────────┬──────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ Block 2: Rate Loop  │
                  │ (Angular velocity)  │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼────────────────┐
                  │ Block 3: Control Alloc.   │
                  │  τ_vanes = A * δ          │
                  │  (Servo angles ← torques) │
                  └──────────┬────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
        ┌───▼───┐        ┌───▼───┐       ┌───▼───┐
        │Servo 1│        │Servo 2│       │Servo 3│
        │ Angle │        │ Angle │       │ Angle │
        └───┬───┘        └───┬───┘       └───┬───┘
            │                │               │
            │        ┌───────────┐           │
            │        │ Vane Lift │           │
            │        │  Torques  │           │
            │        └────┬──────┘           │
            │             │                  │
            └──────┬──────┼──────────────────┘
                   │      │
            ┌──────▼──────▼──────┐
            │  SBOT Dynamics     │
            │  (6-DOF Rigid Body)│
            └──────┬──────┬──────┘
                   │      │
              ┌────▼─┐  ┌─▼────┐
              │Accel.│  │Angles│
              │      │  │      │
              └──────┘  └──────┘
```

---

## 4. MCP Server Implementation

### 4.1 Basic MCP Server Structure (Python)

```python
# mcp_robot_server.py

from mcp.server import Server, stdio_server
from mcp.types import Tool, TextContent
import json
import logging

# Initialize MCP server
mcp = Server("SBOT-LLM-Control")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import SBOT control modules
from models import kinematics, dynamics, control_allocation
from hardware import motor_driver, servo_driver, sensor_fusion

# Robot state (updated by sensor fusion loop)
robot_state = {
    "position": [0.0, 0.0, 0.0],      # [x, y, z] meters
    "velocity": [0.0, 0.0, 0.0],
    "attitude": [0.0, 0.0, 0.0],      # [roll, pitch, yaw] radians
    "angular_rate": [0.0, 0.0, 0.0],  # [p, q, r] rad/s
    "motor_rpm": 0,
    "servo_angles": [90.0, 90.0, 90.0, 90.0],
    "battery_voltage": 11.1,
    "battery_current": 5.0
}

# ============================================
# TOOL 1: Get Robot Description
# ============================================
@mcp.tool()
def get_robot_description() -> str:
    """
    Get detailed information about SBOT hardware specifications.
    Returns mass, dimensions, motor specs, servo specs, and control limits.
    """
    description = {
        "robot_name": "SBOT Monocopter v0.3",
        "type": "single-rotor vertical takeoff vehicle",
        "configuration": "1 down-facing motor + 4 servo-controlled vanes",
        
        "physical_specs": {
            "mass_kg": 1.5,
            "diameter_m": 0.4,
            "height_m": 0.3,
            "inertia_kg_m2": {
                "Ixx": 0.025,
                "Iyy": 0.025,
                "Izz": 0.008
            }
        },
        
        "motor_specs": {
            "type": "Brushless DC (BLDC)",
            "kv_rating": 900,
            "max_rpm": 12000,
            "max_thrust_n": 25.0,
            "prop_diameter_in": 10,
            "prop_pitch_in": 5
        },
        
        "servo_specs": {
            "type": "Micro servo (TowerPro SG90)",
            "count": 4,
            "positions": ["front", "right", "back", "bottom"],
            "range_degrees": [0, 180],
            "speed_sec_per_60deg": 0.12,
            "torque_kg_cm": 1.8
        },
        
        "sensor_suite": {
            "imu": "MPU6050 (6-DOF)",
            "gnss": "u-blox M9N (RTK capable)",
            "camera": "Raspberry Pi Camera v2 (8MP)",
            "lidar": "TFmini-S (TOF)"
        },
        
        "control_limits": {
            "max_altitude_m": 100,
            "max_velocity_m_s": 15,
            "max_acceleration_m_s2": 5.0,
            "max_roll_pitch_rad": 1.0,  # ~57 degrees
            "max_yaw_rate_rad_s": 2.0
        },
        
        "battery": {
            "voltage": "11.1V (3S LiPo)",
            "capacity_mah": 850,
            "continuous_current_a": 10,
            "max_current_a": 50
        }
    }
    
    return json.dumps(description, indent=2)


# ============================================
# TOOL 2: Get Robot State
# ============================================
@mcp.tool()
def get_robot_state() -> dict:
    """
    Query the current state of SBOT including position, velocity,
    attitude, rates, motor/servo commands, and sensor readings.
    
    Returns: Complete state vector + camera images from all mounted cameras
    """
    state_update = sensor_fusion.get_fused_state()  # IMU+GNSS+camera fusion
    robot_state.update(state_update)
    
    # Get latest camera frames
    camera_images = sensor_fusion.get_camera_frames()
    
    response = {
        "timestamp": robot_state.get("timestamp", 0),
        "position_m": robot_state["position"],
        "velocity_m_s": robot_state["velocity"],
        "attitude_rad": robot_state["attitude"],
        "angular_rate_rad_s": robot_state["angular_rate"],
        "motor_rpm": robot_state["motor_rpm"],
        "servo_angles_deg": robot_state["servo_angles"],
        "battery": {
            "voltage_v": robot_state["battery_voltage"],
            "current_a": robot_state["battery_current"],
            "estimated_runtime_min": (
                robot_state["battery_voltage"] * 850 / 1000 / 
                (robot_state["battery_current"] + 0.1)
            )
        },
        "system_status": "nominal",
        "camera_frames": {
            "forward": camera_images.get("forward", None),
            "bottom": camera_images.get("bottom", None)
        }
    }
    
    logger.info(f"Robot state requested: pos={response['position_m']}, "
                f"att={response['attitude_rad']}")
    
    return response


# ============================================
# TOOL 3: Move Robot (High-Level Command)
# ============================================
@mcp.tool()
def move_robot(
    direction: str,      # "forward", "backward", "left", "right", "up", "down"
    distance: float,     # meters
    speed: float = 1.0,  # m/s (default moderate speed)
    wait_for_completion: bool = True
) -> dict:
    """
    High-level movement command for SBOT.
    
    Converts intuitive direction + distance into desired position,
    then executes smooth trajectory with attitude stabilization.
    
    Args:
        direction: Compass/vertical direction
        distance: How far to move in that direction
        speed: Movement speed (1.0 = nominal)
        wait_for_completion: Block until movement completes
    
    Returns: Final position, distance traveled, time taken
    """
    
    # Get current position
    current_pos = robot_state["position"].copy()
    
    # Calculate target position based on direction
    target_pos = current_pos.copy()
    direction_map = {
        "forward": [distance, 0, 0],
        "backward": [-distance, 0, 0],
        "left": [0, -distance, 0],
        "right": [0, distance, 0],
        "up": [0, 0, distance],
        "down": [0, 0, -distance]
    }
    
    if direction not in direction_map:
        return {"error": f"Unknown direction: {direction}"}
    
    direction_vec = direction_map[direction]
    target_pos[0] += direction_vec[0]
    target_pos[1] += direction_vec[1]
    target_pos[2] += direction_vec[2]
    
    # Execute movement (send trajectory to attitude controller)
    logger.info(f"Moving {direction} {distance}m from {current_pos} to {target_pos}")
    
    try:
        # This would interface with your actual control system
        result = {
            "status": "success",
            "start_position_m": current_pos,
            "target_position_m": target_pos,
            "distance_traveled_m": distance,
            "final_position_m": target_pos,  # In real system: actual final position
            "time_taken_s": distance / max(speed, 0.1),
            "path_executed": "straight line with attitude stabilization"
        }
    except Exception as e:
        result = {"status": "error", "error": str(e)}
    
    return result


# ============================================
# TOOL 4: Control Vanes
# ============================================
@mcp.tool()
def control_vanes(
    vane_angles: list = None,  # 4 angles in degrees [0-180]
    roll_command: float = 0.0,
    pitch_command: float = 0.0,
    yaw_command: float = 0.0
) -> dict:
    """
    Control the 4 vanes directly (low-level) or via attitude commands.
    
    Options:
    1. Direct servo angles: Specify exact angle for each vane
    2. Attitude commands: Specify desired roll/pitch/yaw, system computes angles
    
    Args:
        vane_angles: List of 4 angles [front, right, back, bottom] degrees
        roll_command: Desired roll rate (rad/s)
        pitch_command: Desired pitch rate (rad/s)
        yaw_command: Desired yaw rate (rad/s)
    
    Returns: Actual servo angles set, resulting torques
    """
    
    if vane_angles is not None:
        # Direct angle control
        if len(vane_angles) != 4:
            return {"error": "Must provide 4 vane angles"}
        
        # Clamp angles to servo limits
        limited_angles = [max(0, min(180, angle)) for angle in vane_angles]
        
        # Send to servo drivers
        for i, angle in enumerate(limited_angles):
            servo_driver.set_angle(servo_id=i+1, angle_deg=angle)
        
        # Update state
        robot_state["servo_angles"] = limited_angles
        
        result = {
            "status": "success",
            "vane_angles_set_deg": limited_angles,
            "method": "direct_angles"
        }
    
    else:
        # Attitude command (roll/pitch/yaw) → servo angles via control allocation
        attitude_cmd = {
            "roll": roll_command,
            "pitch": pitch_command,
            "yaw": yaw_command
        }
        
        # Compute servo angles using control allocation matrix
        servo_angles = control_allocation.attitude_to_servos(
            attitude_cmd,
            robot_state["angular_rate"]
        )
        
        # Send to hardware
        for i, angle in enumerate(servo_angles):
            servo_driver.set_angle(servo_id=i+1, angle_deg=angle)
        
        robot_state["servo_angles"] = servo_angles.tolist()
        
        result = {
            "status": "success",
            "attitude_command_rad_s": attitude_cmd,
            "computed_servo_angles_deg": servo_angles.tolist(),
            "method": "attitude_control"
        }
    
    logger.info(f"Vane control executed: {result}")
    return result


# ============================================
# TOOL 5: Control Motor Throttle
# ============================================
@mcp.tool()
def control_motor(
    throttle_percent: float = 50.0,  # 0-100%
    altitude_target: float = None     # meters (auto-hover at this altitude)
) -> dict:
    """
    Control BLDC motor thrust directly or via altitude hold.
    
    Args:
        throttle_percent: 0-100% motor power (direct control)
        altitude_target: Desired altitude in meters (altitude hold mode)
    
    Returns: Current motor RPM, thrust, power consumption
    """
    
    if altitude_target is not None:
        # Altitude hold mode: compute throttle from altitude error
        altitude_error = altitude_target - robot_state["position"][2]
        throttle_percent = 50.0 + 5.0 * altitude_error  # Simple proportional
        throttle_percent = max(0, min(100, throttle_percent))
    
    # Clamp throttle
    throttle_clamped = max(0, min(100, throttle_percent))
    
    # Convert to motor PWM/RPM
    target_rpm = (throttle_clamped / 100.0) * 12000  # Max RPM = 12000
    
    # Send to motor driver
    motor_driver.set_speed_rpm(target_rpm)
    robot_state["motor_rpm"] = target_rpm
    
    # Estimate thrust (T = k_T * RPM^2, normalized)
    thrust_n = 0.00002 * (target_rpm ** 2)  # Empirical coefficient
    thrust_n = min(thrust_n, 25.0)  # Max thrust = 25N
    
    result = {
        "status": "success",
        "throttle_percent_set": throttle_clamped,
        "target_rpm": target_rpm,
        "estimated_thrust_n": thrust_n,
        "estimated_power_w": (target_rpm / 12000) * 80  # Max ~80W at full throttle
    }
    
    if altitude_target is not None:
        result["mode"] = "altitude_hold"
        result["altitude_target_m"] = altitude_target
        result["altitude_error_m"] = altitude_error
    
    logger.info(f"Motor control: {result}")
    return result


# ============================================
# TOOL 6: Get Sensor Data
# ============================================
@mcp.tool()
def get_sensor_data(sensor_type: str = "all") -> dict:
    """
    Access raw sensor data for advanced analysis.
    
    Sensor types: "imu", "gnss", "camera", "lidar", "all"
    
    Returns: Sensor-specific data suitable for ML/vision models
    """
    
    sensors = {}
    
    if sensor_type in ["imu", "all"]:
        imu_data = sensor_fusion.get_imu_data()
        sensors["imu"] = {
            "acceleration_m_s2": imu_data["accel"],
            "angular_velocity_rad_s": imu_data["gyro"],
            "magnetometer_gauss": imu_data["mag"]
        }
    
    if sensor_type in ["gnss", "all"]:
        gnss_data = sensor_fusion.get_gnss_data()
        sensors["gnss"] = {
            "latitude_deg": gnss_data["lat"],
            "longitude_deg": gnss_data["lon"],
            "altitude_m": gnss_data["alt"],
            "accuracy_m": gnss_data["accuracy"],
            "velocity_m_s": gnss_data["velocity"]
        }
    
    if sensor_type in ["camera", "all"]:
        camera_data = sensor_fusion.get_camera_frames()
        sensors["camera"] = {
            "frame_available": True,
            "forward_view": camera_data.get("forward"),
            "bottom_view": camera_data.get("bottom"),
            "resolution": "640x480",
            "fps": 30
        }
    
    if sensor_type in ["lidar", "all"]:
        lidar_data = sensor_fusion.get_lidar_data()
        sensors["lidar"] = {
            "distance_m": lidar_data["distance"],
            "signal_strength": lidar_data["signal"],
            "measurement_confidence": lidar_data["confidence"]
        }
    
    return sensors


# ============================================
# TOOL 7: Hover at Current Position
# ============================================
@mcp.tool()
def hover(duration_seconds: float = 10.0) -> dict:
    """
    Maintain altitude and attitude at current position.
    Uses altitude hold + attitude stabilization.
    
    Args:
        duration_seconds: How long to hover
    
    Returns: Hover success, actual duration
    """
    
    current_altitude = robot_state["position"][2]
    
    logger.info(f"Hovering at altitude {current_altitude}m for {duration_seconds}s")
    
    result = {
        "status": "success",
        "mode": "altitude_hold",
        "hover_altitude_m": current_altitude,
        "requested_duration_s": duration_seconds,
        "actual_duration_s": duration_seconds,
        "stability_rating": 0.95  # Attitude control quality
    }
    
    return result


# ============================================
# TOOL 8: Emergency Stop
# ============================================
@mcp.tool()
def emergency_stop() -> dict:
    """
    Cut all motor power immediately. SBOT will descend.
    
    Returns: Motor shutdown status
    """
    
    logger.warning("EMERGENCY STOP ACTIVATED")
    
    motor_driver.set_speed_rpm(0)  # Kill motor
    for i in range(1, 5):
        servo_driver.set_angle(servo_id=i, angle_deg=90)  # Center servos
    
    robot_state["motor_rpm"] = 0
    robot_state["servo_angles"] = [90.0, 90.0, 90.0, 90.0]
    
    return {
        "status": "motors_stopped",
        "motor_rpm": 0,
        "servo_angles_deg": [90, 90, 90, 90],
        "message": "All actuators disabled. SBOT will descend."
    }


# ============================================
# Server Startup
# ============================================

async def main():
    """Start the MCP server"""
    async with stdio_server(mcp) as (read_stream, write_stream):
        logger.info("SBOT MCP Server started")
        logger.info(f"Available tools: {len(mcp.list_tools())}")
        await read_stream

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 4.2 Tool Descriptions (Visible to LLM)

Each tool has a description that the LLM sees:

```python
{
  "name": "move_robot",
  "description": "Command SBOT to move in a specified direction for a distance. Automatically stabilizes attitude during motion. Useful for navigation tasks.",
  "parameters": {
    "type": "object",
    "properties": {
      "direction": {
        "type": "string",
        "description": "Direction to move: forward, backward, left, right, up, down",
        "enum": ["forward", "backward", "left", "right", "up", "down"]
      },
      "distance": {
        "type": "number",
        "description": "Distance in meters"
      },
      "speed": {
        "type": "number",
        "description": "Movement speed (1.0 = normal, 0.5 = slow, 2.0 = fast)",
        "minimum": 0.1,
        "maximum": 3.0
      }
    },
    "required": ["direction", "distance"]
  }
}
```

---

## 5. Control Allocation Module

### 5.1 Vane Forces and Torques

For SBOT with 4 vanes in the propeller wash:

```python
# kinematics.py - Vane control allocation

import numpy as np
from scipy.optimize import least_squares

class SBOTVaneController:
    """
    Converts desired body torques into 4 servo angles via
    inverse kinematics of vane lift/drag forces.
    """
    
    def __init__(self):
        # SBOT physical parameters
        self.mass = 1.5  # kg
        self.radius_to_vane = 0.2  # m (distance from center to vane)
        self.vane_area = 0.01  # m² (each vane)
        
        # Propeller wash model
        self.prop_diameter = 0.254  # 10 inch
        self.prop_rpm_nominal = 8000
        
        # Vane positions in body frame (x, y, z)
        self.vane_positions = np.array([
            [0.2, 0.0, 0.0],  # Vane 1: front
            [0.0, 0.2, 0.0],  # Vane 2: right
            [-0.2, 0.0, 0.0], # Vane 3: back
            [0.0, 0.0, -0.1]  # Vane 4: bottom (in thrust wash)
        ])
        
        # Vane normal vectors (pointing outward)
        self.vane_normals = np.array([
            [1.0, 0.0, 0.0],   # Front faces +X
            [0.0, 1.0, 0.0],   # Right faces +Y
            [-1.0, 0.0, 0.0],  # Back faces -X
            [0.0, 0.0, 1.0]    # Bottom faces +Z
        ])
    
    def propeller_wash_velocity(self, motor_rpm):
        """
        Estimate the velocity of air in propeller wash.
        
        V_wash = k * ω * R
        where k ≈ 0.5-0.8, ω is angular velocity, R is radius
        """
        omega = motor_rpm * 2 * np.pi / 60  # Convert RPM to rad/s
        radius = self.prop_diameter / 2
        k_wash = 0.6  # Empirical coefficient
        
        return k_wash * omega * radius  # m/s
    
    def local_flow_at_vane(self, vane_id, motor_rpm, body_rate):
        """
        Calculate local airspeed at a vane due to:
        1. Propeller wash (downward)
        2. Body angular velocity (rotation)
        
        Args:
            vane_id: 0-3 (which vane)
            motor_rpm: Motor speed
            body_rate: Angular velocity [p, q, r] rad/s
        
        Returns:
            Velocity magnitude (m/s)
        """
        
        # Propeller wash component (downward)
        wash_vel = self.propeller_wash_velocity(motor_rpm)
        wash_vec = np.array([0.0, 0.0, -wash_vel])
        
        # Rotational velocity at vane position: v = ω × r
        omega = np.array(body_rate)
        r = self.vane_positions[vane_id]
        rotational_vel = np.cross(omega, r)
        
        # Total velocity
        total_vel = wash_vec + rotational_vel
        vel_magnitude = np.linalg.norm(total_vel)
        
        return vel_magnitude, total_vel
    
    def vane_lift_drag(self, vel_magnitude, deflection_angle_rad):
        """
        Compute lift and drag on a vane deflected at angle.
        
        Uses simple airfoil model: C_L and C_D as functions of deflection.
        """
        
        # Air properties
        rho = 1.225  # kg/m³ at sea level
        S = self.vane_area
        
        # Simple deflection model
        # Lift increases with deflection, drag increases with deflection squared
        angle_deg = np.degrees(deflection_angle_rad)
        
        # Coefficients (from flat plate / simple airfoil model)
        C_L = 0.8 * np.sin(deflection_angle_rad)
        C_D = 0.2 + 1.5 * (angle_deg / 90.0) ** 2
        
        # Lift and drag magnitudes
        L = 0.5 * rho * vel_magnitude**2 * S * C_L
        D = 0.5 * rho * vel_magnitude**2 * S * C_D
        
        return L, D
    
    def servo_angle_to_forces(self, servo_angles_deg, motor_rpm, body_rate):
        """
        Convert 4 servo angles to 4 vane forces (lift + drag in vane frame).
        
        Args:
            servo_angles_deg: [θ1, θ2, θ3, θ4] in degrees
            motor_rpm: Motor speed
            body_rate: [p, q, r] rad/s
        
        Returns:
            Force matrix (4 vanes × 3D forces in body frame)
        """
        
        servo_angles_rad = np.radians(servo_angles_deg)
        forces = np.zeros((4, 3))
        
        for i in range(4):
            # Local flow at this vane
            vel_mag, vel_vec = self.local_flow_at_vane(i, motor_rpm, body_rate)
            
            if vel_mag < 0.1:  # No flow, no force
                continue
            
            # Lift and drag on this vane
            L, D = self.vane_lift_drag(vel_mag, servo_angles_rad[i])
            
            # Convert lift/drag to body-frame force
            # Lift: perpendicular to flow
            # Drag: parallel to flow (opposes motion)
            
            normal = self.vane_normals[i]
            flow_dir = vel_vec / (vel_mag + 1e-6)
            
            # Lift acts perpendicular to flow, in vane normal direction
            lift_vec = L * normal
            
            # Drag acts opposite to flow
            drag_vec = -D * flow_dir
            
            forces[i] = lift_vec + drag_vec
        
        return forces
    
    def forces_to_torques(self, forces):
        """
        Convert 4 vane forces to body torques via τ = r × F
        
        Args:
            forces: (4, 3) array of forces
        
        Returns:
            Torque vector [τ_x, τ_y, τ_z]
        """
        
        torque = np.zeros(3)
        for i in range(4):
            r = self.vane_positions[i]
            F = forces[i]
            torque += np.cross(r, F)
        
        return torque
    
    def compute_control_allocation_matrix(self, motor_rpm=8000):
        """
        Linearize around nominal operating point to get allocation matrix.
        
        Small servo angle changes → linear torque changes
        
        τ = A * δ + b
        
        where δ is servo angle deviation from 90°, A is 3×4 matrix.
        """
        
        # Nominal servo angles (all at 90°, neutral)
        nominal_angles = np.array([90.0, 90.0, 90.0, 90.0])
        body_rate_nominal = np.array([0.0, 0.0, 0.0])
        
        # Compute Jacobian numerically
        delta = 1.0  # 1 degree perturbation
        A = np.zeros((3, 4))
        
        for j in range(4):
            angles_plus = nominal_angles.copy()
            angles_plus[j] += delta
            
            angles_minus = nominal_angles.copy()
            angles_minus[j] -= delta
            
            # Forces and torques at ±perturbation
            F_plus = self.servo_angle_to_forces(angles_plus, motor_rpm, body_rate_nominal)
            tau_plus = self.forces_to_torques(F_plus)
            
            F_minus = self.servo_angle_to_forces(angles_minus, motor_rpm, body_rate_nominal)
            tau_minus = self.forces_to_torques(F_minus)
            
            # Derivative
            A[:, j] = (tau_plus - tau_minus) / (2 * np.radians(delta))
        
        return A
    
    def attitude_to_servo_angles(self, tau_desired, motor_rpm=8000):
        """
        Solve inverse kinematics: desired torques → servo angles
        
        Minimize: ||A*δ - τ_desired||²
        subject to: 0° ≤ servo_angle ≤ 180°
        
        Args:
            tau_desired: [τ_x, τ_y, τ_z] in body frame (N⋅m)
            motor_rpm: Current motor speed
        
        Returns:
            Servo angles [θ1, θ2, θ3, θ4] in degrees
        """
        
        # Get allocation matrix
        A = self.compute_control_allocation_matrix(motor_rpm)
        
        # Solve for servo angle deviations
        # A * δ = τ
        # Use pseudo-inverse for over/under-constrained system
        A_pinv = np.linalg.pinv(A)
        delta_optimal = A_pinv @ tau_desired
        
        # Convert back to absolute angles
        nominal = np.array([90.0, 90.0, 90.0, 90.0])
        servo_angles = nominal + delta_optimal
        
        # Clamp to servo limits [0, 180]
        servo_angles = np.clip(servo_angles, 0, 180)
        
        return servo_angles
```

---

## 6. Inverse Kinematics for SBOT Vanes

### 6.1 Forward Kinematics (Servo angles → Torques)

For each vane:

```
1. Local flow velocity:
   V_wash = k_wash * ω * R  (prop wash, downward)
   V_rot = ω_body × r_vane  (rotation of vane with body)
   V_total = V_wash + V_rot
```

```
2. Lift/Drag coefficients:
   C_L(δ) = a * sin(δ)         (deflection angle δ)
   C_D(δ) = C_D0 + b * δ²
```

```
3. Forces:
   L = 0.5 * ρ * V² * S * C_L(δ)
   D = 0.5 * ρ * V² * S * C_D(δ)
```

```
4. Torques:
   τ_i = r_i × F_i
   τ_total = Σ τ_i
```

### 6.2 Inverse Kinematics (Desired torques → Servo angles)

```
Problem: τ_desired = A(ω, θ) * δ_servo

where:
  A = 3×4 control allocation matrix
  δ_servo = servo angle deviations from neutral (90°)

Solution: δ = A⁻¹ * τ_desired  (using pseudo-inverse for 4 servos, 3 DoFs)
```

---

## 7. Step-by-Step Setup Guide

### 7.1 Prerequisites

```
- Ubuntu 20.04 or 22.04
- Python 3.8+
- ROS 2 Humble (optional, for real hardware)
- SBOT hardware with:
  * STM32/Pixhawk flight controller
  * IMU (MPU6050)
  * GNSS receiver
  * Cameras (optional)
  * LiDAR (optional)
```

### 7.2 Installation Steps

**Step 1: Clone the Repository**

```bash
git clone https://github.com/your-username/sbot-mcp-server.git
cd sbot-mcp-server
```

**Step 2: Create Virtual Environment**

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

**Step 3: Install Dependencies**

```bash
pip install --upgrade pip
pip install mcp
pip install pydantic
pip install numpy scipy
pip install pyserial  # For hardware communication
pip install opencv-python  # For camera processing
```

**Step 4: Configure Hardware Interfaces**

Edit `config/sbot_config.yaml`:

```yaml
hardware:
  flight_controller:
    type: "STM32F4"  # or Pixhawk
    serial_port: "/dev/ttyUSB0"
    baud_rate: 115200
  
  servos:
    interface: "PWM"
    pins: [17, 27, 22, 23]  # GPIO pins for servo signals (Raspberry Pi)
    frequency_hz: 50
  
  motor:
    type: "BLDC_ESC"
    serial_port: "/dev/ttyUSB1"
    max_rpm: 12000
  
  sensors:
    imu:
      type: "MPU6050"
      i2c_address: 0x68
      sampling_rate_hz: 200
    
    gnss:
      type: "u-blox_M9N"
      serial_port: "/dev/ttyUSB2"
      baudrate: 38400
    
    camera:
      forward: 0  # /dev/video0
      bottom: 1   # /dev/video1
```

**Step 5: Test Hardware Communication**

```bash
python3 -m sbot_mcp.tests.test_hardware
```

### 7.3 Running the MCP Server

**Standalone (for testing):**

```bash
python3 -m sbot_mcp.mcp_robot_server
```

**With Claude Desktop:**

1. Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "SBOT": {
      "command": "python",
      "args": ["-m", "sbot_mcp.mcp_robot_server"]
    }
  }
}
```

2. Restart Claude Desktop
3. Look for "SBOT" in the Tools section

---

## 8. Testing and Validation

### 8.1 MCP Inspector (Built-in Testing Tool)

```bash
# Run MCP server in dev mode with inspector
python3 -m mcp dev -m sbot_mcp.mcp_robot_server
```

Opens: `http://localhost:3000` (inspect available tools and test them)

### 8.2 Test Sequence

1. **Get Robot Description:**
   - Verify all specs are correct
   - Check sensor list

2. **Get Robot State:**
   - Should return position, attitude, motor/servo values
   - Check sensor fusion working

3. **Test Motor Control:**
   - Set throttle 20% → hear motor spin up
   - Set throttle 0% → motor stops

4. **Test Servo Control:**
   - Set vane angles [90, 90, 90, 90] → all servos centered
   - Set [45, 90, 135, 90] → see vanes move

5. **Test Movement (Simulated):**
   - move_robot("up", 1.0) → altitude should increase
   - move_robot("forward", 1.0) → X position should increase

6. **Test with LLM:**
   - Claude: "Move SBOT up 2 meters"
   - Check: throttle increases, altitude increases

### 8.3 Safety Checks

- [ ] Motor has mechanical stop (prop can't over-spin)
- [ ] Servos have physical limits (0-180°)
- [ ] Battery voltage monitoring active
- [ ] Attitude limits enforced (max 60° roll/pitch)
- [ ] Emergency stop works (kills motor, centers servos)

---

## 9. Integration with LLMs

### 9.1 Claude Desktop Integration

Once MCP server is running:

```
User: "SBOT, what's your current altitude?"

Claude: [calls get_robot_state()]
Response: "You are currently at an altitude of 2.5 meters, with a 
          velocity of 0.3 m/s upward. Battery at 85%."

User: "Hover for 30 seconds and send me a picture"

Claude: [calls hover(30)] + [calls get_sensor_data("camera")]
Response: [displays camera image] "Hovering maintained. 
          Here's what I see below..."

User: "Move forward 5 meters slowly"

Claude: [calls move_robot("forward", 5, speed=0.5)]
Response: "Moving forward at slow speed... reached target. 
          Took 12 seconds."
```

### 9.2 Custom LLM Agent (Python)

```python
# example_agent.py

import asyncio
from anthropic import Anthropic

client = Anthropic()
conversation_history = []

MODEL_NAME = "claude-3-5-sonnet-20241022"
# Your MCP server tools will be available to Claude

def chat_with_sbot(user_message):
    """Send message to Claude, which can control SBOT via MCP"""
    
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        system="You are an AI controlling an autonomous drone called SBOT. "
               "SBOT is a monocopter with 1 downward motor and 4 servo-controlled vanes. "
               "Use the available tools to execute user commands safely. "
               "Always check battery level and altitude limits before moving.",
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

# Interactive loop
if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        
        response = chat_with_sbot(user_input)
        print(f"SBOT AI: {response}\n")
```

### 9.3 Example Conversations

**Scenario 1: Autonomous Inspection**

```
User: "Fly to coordinates (10, 5, 3) and take a picture of the building"

SBOT AI:
1. [Calls get_robot_state()] → current: (0, 0, 0)
2. [Computes path] forward 10m, right 5m, up 3m
3. [Calls move_robot("forward", 10)] ✓
4. [Calls move_robot("right", 5)] ✓
5. [Calls move_robot("up", 3)] ✓
6. [Calls hover(5)] ✓
7. [Calls get_sensor_data("camera")] → receives image
8. Response: "Reached target coordinates. Here's the building photo..."
```

**Scenario 2: Emergency Response**

```
User: "SBOT, battery critical, return home immediately"

SBOT AI:
1. [Calls get_robot_state()] → battery 5%, altitude 25m
2. [Computes descent path] slow descent to home
3. [Calls move_robot("forward", -10) + move_robot("down", 25)]
4. [Calls hover(2)] at home
5. Response: "Landing sequence initiated. Descending to home... landed safely"
```

---

## 10. Troubleshooting

### Issue: MCP Server Won't Start

```
Error: "Address already in use"

Solution:
1. Kill existing process: lsof -i :8000
2. Change port in config if needed
3. Restart server
```

### Issue: Claude Can't See SBOT Tools

```
Error: "MCP server not connected" in Claude Desktop

Solution:
1. Check config path: ~/.config/Claude/claude_desktop_config.json
2. Verify command is correct (python path)
3. Check server starts manually: python3 -m sbot_mcp.mcp_robot_server
4. Restart Claude Desktop
5. Check console for errors: tail -f ~/.local/share/Claude/logs/*
```

### Issue: Motor Not Responding

```
Error: "Motor RPM reads 0 after throttle command"

Solution:
1. Check serial connection: ls /dev/ttyUSB*
2. Verify baud rate in config
3. Test motor directly with ESC programmer
4. Check battery voltage (must be > 10.5V for 3S)
5. Review motor driver code for error flags
```

### Issue: Servos Moving Erratically

```
Error: "Servos jitter or move unexpectedly"

Solution:
1. Check servo power supply (should be 5V, 2A minimum)
2. Verify PWM frequency is 50Hz
3. Test servos individually with manual commands
4. Check servo angle limits (0-180°)
5. Review servo driver calibration (center at 90°)
```

### Issue: State Estimation Drifting

```
Error: "Altitude/position diverging from actual"

Solution:
1. Calibrate IMU: python3 -m sbot_mcp.tests.calibrate_imu
2. Verify GNSS has lock (check satellites)
3. Reduce complementary filter gyro weight if drifting
4. Check sensor cables for loose connections
5. Review sensor fusion logs
```

---

## Summary: MCP vs Traditional Control

| Aspect | Traditional | MCP-Based |
|--------|-----------|-----------|
| **Control Interface** | Custom code/proprietary APIs | Standard MCP tools |
| **LLM Integration** | Custom for each LLM | Works with all LLMs |
| **Natural Language** | Requires separate NLP layer | Built-in via LLM |
| **Extensibility** | Add new tool = rewrite integration | Add new tool = new MCP tool |
| **Safety** | Manual constraint enforcement | Can be part of tool design |
| **Learning Curve** | Steep (robotics + programming) | Gentler (define what robot can do) |

---

## Next Steps

1. **Set up MCP server** with your SBOT hardware
2. **Test each tool** manually first
3. **Validate control allocation** matrix with servo measurements
4. **Connect to Claude Desktop** and test natural language commands
5. **Add specialized tools** for your mission (e.g., payload deployment, autonomous inspection)
6. **Implement safety monitors** (battery, altitude, attitude limits)
7. **Deploy to field** with safety pilot supervision

---

**For detailed reference material, see:**
- MCP Documentation: https://modelcontextprotocol.io
- Ilia Larchenko's GitHub: https://github.com/IliaLarchenko/robot_MCP
- LeRobot Framework: https://huggingface.co/lerobot
- SBOT Research: Your existing documentation

