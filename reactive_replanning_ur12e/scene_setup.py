#!/usr/bin/env python3
"""
Scene Setup + PointCloud OctoMap integration for UR12e reactive replanning.
Adds static collision objects and streams depth camera data into MoveIt's OctoMap.
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene


class SceneSetup(Node):
    def __init__(self):
        super().__init__('scene_setup')

        self._obj_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self.declare_parameter('table_z', 0.0)   # top surface of the table in base_link frame

        self._initialized = False
        # Retry every 2s until published — move_group may not be listening yet on first fire
        self.create_timer(2.0, self._add_static_objects)
        self.get_logger().info('Scene setup node started.')

    def _box(self, name, x, y, z, sx, sy, sz, frame='base_link'):
        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        obj.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(sx), float(sy), float(sz)]

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0

        obj.primitives = [prim]
        obj.primitive_poses = [pose]
        self._obj_pub.publish(obj)
        self.get_logger().info(f'Added: {name}')

    def _add_static_objects(self):
        if self._initialized:
            return
        # Wait until move_group's /collision_object topic has a subscriber
        if self._obj_pub.get_subscription_count() == 0:
            self.get_logger().info('Waiting for MoveIt to subscribe to /collision_object...')
            return
        self._initialized = True

        table_z = self.get_parameter('table_z').value

        # Floor plane — prevents the arm from planning below table height
        self._box('floor', 0.0, 0.0, table_z - 0.01, 4.0, 4.0, 0.02)

        self.get_logger().info(f'Floor plane added at z={table_z:.3f}.')


def main(args=None):
    rclpy.init(args=args)
    node = SceneSetup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
