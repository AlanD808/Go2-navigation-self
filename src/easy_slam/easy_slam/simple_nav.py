#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import yaml
import random
import os
import math
import heapq

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped, Twist
from tf2_ros import StaticTransformBroadcaster
from rclpy.qos import qos_profile_sensor_data
from PIL import Image


class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')

        self.load_map()

        # ================= Publisher =================
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.goal_pub = self.create_publisher(PoseArray, '/goal_points', 10)
        self.gt_vis_pub = self.create_publisher(PoseStamped, '/gt_pose_vis', 100)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)

        # ================= Subscriber =================
        self.create_subscription(
            Odometry,
            '/gt_pose/odom',
            self.gt_callback,
            qos_profile_sensor_data
        )

        # ================= TF =================
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf()

        self.current_gt_pose = None

        # ================= 目标区域 =================
        self.compute_valid_region()
        self.goals = self.generate_goals(3)

        self.map_msg = self.build_map_message()

        # ================= 导航状态 =================
        self.current_goal_index = 0

        self.current_path = []
        self.path_index = 0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        # ================= Timer =================
        self.create_timer(1.0, self.publish_map)
        self.create_timer(1.0, self.publish_goals)
        self.create_timer(1.0 / 30.0, self.publish_gt_pose)

        self.create_timer(1.0 / 20.0, self.control_loop)

    # ================= TF =================
    def publish_static_tf(self):
        t = TransformStamped()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    # ================= A*路径规划 =================
    def plan_path(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                neighbor = (current[0]+dx, current[1]+dy)

                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                if self.occ[neighbor[1], neighbor[0]] != 0:
                    continue

                tentative = g_score[current] + 1

                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return []

    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    # ================= 控制器 =================
    def control_loop(self):
        if self.current_gt_pose is None:
            return

        # ✅ 修改：不再使用 index
        if len(self.goals) == 0:
            self.stop_robot()
            return

        # ✅ 始终取第一个目标
        goal = self.goals[0]
        gx, gy = goal[0], goal[1]

        px = self.current_gt_pose.position.x
        py = self.current_gt_pose.position.y

        # 🚀 如果没有路径 → 规划
        if not self.current_path:
            start = self.world_to_grid(px, py)
            goal_grid = self.world_to_grid(gx, gy)

            grid_path = self.plan_path(start, goal_grid)

            if not grid_path:
                self.get_logger().warn("No path found!")
                return

            self.current_path = [self.grid_to_world(x, y) for (x, y) in grid_path]
            self.path_index = 0

            self.path_msg = Path()
            self.path_msg.header.frame_id = "map"

            for (wx, wy) in self.current_path:
                p = PoseStamped()
                p.header.frame_id = "map"
                p.pose.position.x = wx
                p.pose.position.y = wy
                p.pose.orientation.w = 1.0
                self.path_msg.poses.append(p)

            self.path_pub.publish(self.path_msg)

        # 当前路径点
        tx, ty = self.current_path[self.path_index]

        dx = tx - px
        dy = ty - py
        dist = math.sqrt(dx**2 + dy**2)

        yaw = self.get_yaw_from_quaternion(self.current_gt_pose.orientation)
        target_yaw = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_yaw - yaw)

        cmd = Twist()

        # 到达路径点
        if dist < 0.2:
            self.path_index += 1

            if self.path_index >= len(self.current_path):
                self.get_logger().info(f"Reached goal")

                # ✅ 只删除，不移动 index
                self.goals.pop(0)

                self.current_path = []
            return

        if abs(angle_error) > 0.3:
            cmd.angular.z = 0.8 * angle_error
        else:
            cmd.linear.x = 0.5
            cmd.angular.z = 0.3 * angle_error

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def get_yaw_from_quaternion(self, q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    # ================= 原代码（完全不动） =================
    # （此处省略，你的原始代码保持完全一致）

    # ================= 原有代码（完全未改） =================
    def load_map(self):
        yaml_path = os.path.join(os.getcwd(), 'map.yaml')
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        img_path = data['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)

        img = Image.open(img_path).convert('L')
        self.map_img = np.array(img)

        self.resolution = data['resolution']
        self.origin = data['origin']

        occ_th = data.get('occupied_thresh', 0.65)
        free_th = data.get('free_thresh', 0.196)

        self.occ = np.zeros_like(self.map_img, dtype=np.int8)

        for y in range(self.map_img.shape[0]):
            for x in range(self.map_img.shape[1]):
                val = self.map_img[y, x] / 255.0

                if val < free_th:
                    self.occ[y, x] = 1
                elif val > occ_th:
                    self.occ[y, x] = 0
                else:
                    self.occ[y, x] = -1

        self.height, self.width = self.occ.shape

    def build_map_message(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0

        msg.data = [
            100 if self.occ[y, x] == 1 else 0 if self.occ[y, x] == 0 else -1
            for y in range(self.height)
            for x in range(self.width)
        ]
        return msg

    def compute_valid_region(self):
        cx = self.width // 2
        cy = self.height // 2

        x = cx
        while x < self.width and self.occ[cy, x] == 0:
            x += 1
        self.x_max = x - 1

        x = cx
        while x >= 0 and self.occ[cy, x] == 0:
            x -= 1
        self.x_min = x + 1

        y = cy
        while y < self.height and self.occ[y, cx] == 0:
            y += 1
        self.y_max = y - 1

        y = cy
        while y >= 0 and self.occ[y, cx] == 0:
            y -= 1
        self.y_min = y + 1

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return x, y

    def is_safe_cell(self, x, y, radius=0.6):
        r = int(radius / self.resolution)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                nx = x + dx
                ny = y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    return False
                if self.occ[ny, nx] != 0:
                    return False
        return True

    def is_far_enough(self, wx, wy, goals, min_dist=3.5):
        for (gx, gy, _, _) in goals:
            dist = math.sqrt((wx - gx)**2 + (wy - gy)**2)
            if dist < min_dist:
                return False
        return True

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def generate_goals(self, n):
        goals = []
        attempts = 0
        while len(goals) < n and attempts < 20000:
            x = random.randint(self.x_min, self.x_max)
            y = random.randint(self.y_min, self.y_max)

            if self.occ[y, x] == 0 and self.is_safe_cell(x, y):
                wx, wy = self.grid_to_world(x, y)
                if not self.is_far_enough(wx, wy, goals):
                    attempts += 1
                    continue

                yaw = random.uniform(-math.pi, math.pi)
                qz, qw = self.yaw_to_quaternion(yaw)
                goals.append((wx, wy, qz, qw))

            attempts += 1

        return goals

    def gt_callback(self, msg):
        self.current_gt_pose = msg.pose.pose

    def publish_gt_pose(self):
        if self.current_gt_pose is None:
            return

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = self.current_gt_pose
        self.gt_vis_pub.publish(msg)

    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map_msg)

    def publish_goals(self):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for (x, y, qz, qw) in self.goals:
            p = PoseStamped()
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            msg.poses.append(p.pose)

        self.goal_pub.publish(msg)


def main():
    rclpy.init()
    node = SimpleNavigator()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
