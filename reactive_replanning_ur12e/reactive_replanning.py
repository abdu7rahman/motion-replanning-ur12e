#!/usr/bin/env python3
"""
Reactive Replanning for UR12e + Robotiq Hand-E
Adapted from ME5250 UR5 project by Abdul Rahman.

Strategy: Exploit kinematic redundancy — compute 30 IK solutions for the
goal pose, rank by manipulability, and iterate through them when an obstacle
blocks the current trajectory.
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK, GetPlanningScene
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

import numpy as np
import time
import threading
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class IKSolution:
    joint_positions: List[float]
    manipulability: float
    is_valid: bool


class ReactiveReplannerUR12e(Node):
    def __init__(self):
        super().__init__('reactive_replanner_ur12e')

        self.callback_group = ReentrantCallbackGroup()

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        self.declare_parameter('pick_x',  0.45)
        self.declare_parameter('pick_y',  0.0)
        self.declare_parameter('pick_z',  0.35)
        self.declare_parameter('place_x', 0.0)
        self.declare_parameter('place_y', 0.55)
        self.declare_parameter('place_z', 0.35)
        self.declare_parameter('ik_attempts', 30)
        self.declare_parameter('vel_scale', 0.3)
        self.declare_parameter('acc_scale', 0.3)

        self.current_joint_state = None
        self.obstacle_detected = False

        self.pick_pose  = self._make_pose(
            self.get_parameter('pick_x').value,
            self.get_parameter('pick_y').value,
            self.get_parameter('pick_z').value,
            0.0, 1.0, 0.0, 0.0,  # tool pointing down
        )
        self.place_pose = self._make_pose(
            self.get_parameter('place_x').value,
            self.get_parameter('place_y').value,
            self.get_parameter('place_z').value,
            0.0, 1.0, 0.0, 0.0,
        )

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10,
            callback_group=self.callback_group)

        self.collision_sub = self.create_subscription(
            CollisionObject, '/collision_object', self._collision_cb, 10,
            callback_group=self.callback_group)

        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.callback_group)
        self.scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene', callback_group=self.callback_group)

        self.move_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self.callback_group)
        self.exec_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.callback_group)
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_action_controller/gripper_cmd',
            callback_group=self.callback_group)

        self.get_logger().info('Waiting for services...')
        self.ik_client.wait_for_service()
        self.scene_client.wait_for_service()
        self.move_client.wait_for_server()
        self.exec_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('All services ready — UR12e reactive replanner online.')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_pose(self, x, y, z, qx, qy, qz, qw) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.x, p.orientation.y = float(qx), float(qy)
        p.orientation.z, p.orientation.w = float(qz), float(qw)
        return p

    def _js_cb(self, msg: JointState):
        self.current_joint_state = msg

    def _collision_cb(self, msg: CollisionObject):
        if msg.operation == CollisionObject.ADD and (
            'obstacle' in msg.id.lower() or 'dynamic' in msg.id.lower()
        ):
            self.get_logger().warn(f'OBSTACLE DETECTED: {msg.id}')
            self.obstacle_detected = True

    def _current_joints(self) -> List[float]:
        if self.current_joint_state is None:
            return [0.0] * 6
        out = []
        for name in self.joint_names:
            try:
                idx = list(self.current_joint_state.name).index(name)
                out.append(self.current_joint_state.position[idx])
            except (ValueError, IndexError):
                out.append(0.0)
        return out

    def _random_seed(self) -> List[float]:
        limits = [(-3.14, 3.14), (-2.5, 0.0), (-2.0, 2.0),
                  (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]
        return [np.random.uniform(lo, hi) for lo, hi in limits]

    def _manipulability(self, joints: List[float]) -> float:
        score = 1.0
        for j in joints:
            score *= min(abs(j - 3.14), abs(j + 3.14)) / 3.14
        return score

    # ── IK ────────────────────────────────────────────────────────────────────

    def compute_ik_solutions(self, target: Pose, num: int = 30) -> List[IKSolution]:
        solutions, seen = [], set()
        self.get_logger().info(f'Computing {num} IK solutions...')

        for i in range(num):
            seed = self._current_joints() if i == 0 else self._random_seed()

            req = GetPositionIK.Request()
            req.ik_request.group_name = 'ur_manipulator'
            req.ik_request.robot_state.joint_state.name = self.joint_names
            req.ik_request.robot_state.joint_state.position = seed
            req.ik_request.pose_stamped.header.frame_id = 'base_link'
            req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
            req.ik_request.pose_stamped.pose = target
            req.ik_request.timeout.nanosec = 200_000_000
            req.ik_request.avoid_collisions = True

            fut = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

            if fut.result() and fut.result().error_code.val == MoveItErrorCodes.SUCCESS:
                pos = []
                for name in self.joint_names:
                    try:
                        idx = list(fut.result().solution.joint_state.name).index(name)
                        pos.append(fut.result().solution.joint_state.position[idx])
                    except Exception:
                        break
                if len(pos) == 6:
                    key = tuple(round(p, 1) for p in pos)
                    if key not in seen:
                        seen.add(key)
                        solutions.append(IKSolution(pos, self._manipulability(pos), True))

        solutions.sort(key=lambda s: s.manipulability, reverse=True)
        self.get_logger().info(f'Found {len(solutions)} unique IK solutions')
        return solutions

    # ── planning / execution ──────────────────────────────────────────────────

    def plan_to_joints(self, target: List[float]) -> Optional[JointTrajectory]:
        goal = MoveGroup.Goal()
        goal.request.group_name = 'ur_manipulator'
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor     = self.get_parameter('vel_scale').value
        goal.request.max_acceleration_scaling_factor = self.get_parameter('acc_scale').value

        c = Constraints()
        for name, pos in zip(self.joint_names, target):
            jc = JointConstraint()
            jc.joint_name, jc.position = name, pos
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)

        fut = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.result() or not fut.result().accepted:
            return None

        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=30.0)
        res = res_fut.result()
        if res.result.error_code.val == MoveItErrorCodes.SUCCESS:
            return res.result.planned_trajectory.joint_trajectory
        return None

    def execute_trajectory(self, traj: JointTrajectory) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = traj
        self.obstacle_detected = False

        fut = self.exec_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if not fut.result() or not fut.result().accepted:
            return False

        gh = fut.result()
        res_fut = gh.get_result_async()

        while not res_fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.obstacle_detected:
                self.get_logger().warn('OBSTACLE — aborting trajectory')
                gh.cancel_goal_async()
                time.sleep(0.5)
                return False

        return res_fut.result().result.error_code.val == MoveItErrorCodes.SUCCESS

    def gripper(self, open: bool):
        goal = GripperCommand.Goal()
        goal.command.position   = 0.025 if open else 0.0
        goal.command.max_effort = 50.0
        fut = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result():
            fut.result().get_result_async()
        time.sleep(0.5)

    # ── core replanning logic ─────────────────────────────────────────────────

    def move_with_replanning(self, pose: Pose, label: str) -> bool:
        self.get_logger().info(f'Moving to {label}...')
        solutions = self.compute_ik_solutions(pose, self.get_parameter('ik_attempts').value)

        if not solutions:
            self.get_logger().error(f'No IK solutions for {label}')
            return False

        traj = self.plan_to_joints(solutions[0].joint_positions)
        if traj is None:
            self.get_logger().error('Initial planning failed')
            return False

        if self.execute_trajectory(traj):
            return True

        if not self.obstacle_detected:
            return False

        self.get_logger().info('=' * 50)
        self.get_logger().info('REPLANNING via kinematic redundancy')
        self.get_logger().info(f'Testing {len(solutions) - 1} alternative configs...')
        self.get_logger().info('=' * 50)

        for i, sol in enumerate(solutions[1:], 2):
            self.get_logger().info(
                f'Config {i}/{len(solutions)} (manip={sol.manipulability:.4f})')
            self.obstacle_detected = False

            traj = self.plan_to_joints(sol.joint_positions)
            if traj is None:
                self.get_logger().info(f'  → planning blocked')
                continue

            if self.execute_trajectory(traj):
                self.get_logger().info(f'  → SUCCESS with config {i}')
                return True

        self.get_logger().error('All configurations exhausted — giving up')
        return False

    # ── demo ─────────────────────────────────────────────────────────────────

    def run_demo(self):
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('UR12e + Hand-E Reactive Replanning Demo')
        self.get_logger().info('=' * 60)
        self.get_logger().info('During motion, inject an obstacle:')
        self.get_logger().info('  ros2 run reactive_replanning_ur12e insert_obstacle.py --x 0.3 --y 0.2 --z 0.4')
        self.get_logger().info('')
        self.get_logger().info('Press ENTER to start...')
        input()

        while self.current_joint_state is None:
            self.get_logger().info('Waiting for joint states...')
            rclpy.spin_once(self, timeout_sec=1.0)

        for cycle in range(3):
            self.get_logger().info(f'\n===== CYCLE {cycle + 1}/3 =====')

            self.gripper(open=True)
            if self.move_with_replanning(self.pick_pose, 'PICK'):
                self.get_logger().info('At pick — closing gripper')
                self.gripper(open=False)
                time.sleep(0.5)

                if self.move_with_replanning(self.place_pose, 'PLACE'):
                    self.get_logger().info('At place — opening gripper')
                    self.gripper(open=True)
                    time.sleep(0.5)
            else:
                self.get_logger().warn('Failed to reach pick position')

            self.obstacle_detected = False

        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('DEMO COMPLETE')
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveReplannerUR12e()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=node.run_demo)
    thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
