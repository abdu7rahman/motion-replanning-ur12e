#!/usr/bin/env python3
"""
Scene Setup + PointCloud OctoMap integration for UR12e reactive replanning.
Adds static collision objects and streams depth camera data into MoveIt's OctoMap.
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene


class SceneSetup(Node):
    def __init__(self):
        super().__init__('scene_setup')

        self._obj_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self._pc_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self._pc_cb, 10)

        self._initialized = False
        self._last_update = self.get_clock().now()

        self.create_timer(2.0, self._add_static_objects)
        self.get_logger().info('Scene setup node started — waiting for camera...')

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
        self._initialized = True

        # Floor plane
        self._box('floor', 0.0, 0.0, -0.01, 4.0, 4.0, 0.02)

        # Work surface (adjust z to match your table height)
        self._box('work_table', 0.45, 0.0, -0.15, 0.8, 0.8, 0.3)

        # Pick zone objects (update to match your actual setup)
        self._box('pick_zone_left',   0.4,  0.15, 0.02, 0.06, 0.06, 0.06)
        self._box('pick_zone_center', 0.45, 0.0,  0.02, 0.06, 0.06, 0.06)
        self._box('pick_zone_right',  0.4, -0.15, 0.02, 0.06, 0.06, 0.06)

        self.get_logger().info('Static scene objects added.')

    def _pc_cb(self, msg: PointCloud2):
        now = self.get_clock().now()
        elapsed = (now - self._last_update).nanoseconds / 1e9
        if elapsed < 1.0:
            return
        self._last_update = now
        # OctoMap updater (sensors_3d plugin) handles the actual voxelization.
        # This callback is a hook for custom obstacle detection if needed.


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
