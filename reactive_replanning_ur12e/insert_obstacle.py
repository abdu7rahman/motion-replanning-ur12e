#!/usr/bin/env python3
"""
Inject or remove a dynamic obstacle in the MoveIt planning scene.
Use during execution to trigger reactive replanning.

Usage:
  ros2 run reactive_replanning_ur12e insert_obstacle.py --x 0.3 --y 0.2 --z 0.4
  ros2 run reactive_replanning_ur12e insert_obstacle.py --name obstacle --remove
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import argparse
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject


class ObstacleInserter(Node):
    def __init__(self):
        super().__init__('obstacle_inserter')
        self._pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        time.sleep(0.5)

    def add(self, name, x, y, z, radius=0.05, height=0.3):
        obj = CollisionObject()
        obj.header.frame_id = 'base_link'
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        obj.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [float(height), float(radius)]

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        pose.orientation.w = 1.0

        obj.primitives = [prim]
        obj.primitive_poses = [pose]

        for _ in range(5):
            self._pub.publish(obj)
            time.sleep(0.1)
        self.get_logger().info(f'Added obstacle "{name}" at ({x}, {y}, {z})')

    def remove(self, name):
        obj = CollisionObject()
        obj.header.frame_id = 'base_link'
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        obj.operation = CollisionObject.REMOVE
        for _ in range(5):
            self._pub.publish(obj)
            time.sleep(0.1)
        self.get_logger().info(f'Removed obstacle "{name}"')


def main():
    _sig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',   default='obstacle')
    parser.add_argument('--x',      type=float, default=0.3)
    parser.add_argument('--y',      type=float, default=0.2)
    parser.add_argument('--z',      type=float, default=0.4)
    parser.add_argument('--radius', type=float, default=0.05)
    parser.add_argument('--height', type=float, default=0.3)
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    node = ObstacleInserter()
    try:
        if args.remove:
            node.remove(args.name)
        else:
            node.add(args.name, args.x, args.y, args.z, args.radius, args.height)
        rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
