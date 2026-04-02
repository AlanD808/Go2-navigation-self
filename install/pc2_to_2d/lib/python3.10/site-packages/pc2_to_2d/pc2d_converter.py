#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import message_filters
import numpy as np
import transforms3d.euler as t3e
import math
from collections import defaultdict
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster  # --- TF STATIC ADDED ---

class PC2DConverter(Node):
    def __init__(self):
        super().__init__('pc2d_converter')

        # ---------- 参数 ----------
        self.horizontal_beams = 440
        self.min_range = 0.5
        self.max_range = 130.0
        self.angle_min = -math.pi
        self.angle_max = math.pi

        # ---------- 高度参数 ----------
        self.slice_z = 0.4
        self.z_band = 0.2

        # ---------- XY聚类参数 ----------
        self.grid_size = 0.05   # ⭐ 5cm网格
        self.min_points_per_cell = 2  # ⭐ 至少2个点

        self.frame_count = 0

        self.laser_pub = self.create_publisher(LaserScan, '/velodyne_2d', 10)
        self.get_logger().info("Publisher /velodyne_2d initialized.")

        # --- TF STATIC ADDED ---
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = 'base_link'
        static_tf.child_frame_id = 'velodyne_2d'
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.x = 0.0
        static_tf.transform.rotation.y = 0.0
        static_tf.transform.rotation.z = 0.0
        static_tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(static_tf)
        self.get_logger().info("Static transform base_link -> velodyne_2d published.")
        self.get_logger().info("Static transform base_link -> velodyne_2d published.")
        # --- END TF STATIC ADDED ---

        pc_sub = message_filters.Subscriber(self, PointCloud2, '/velodyne_points')
        gt_sub = message_filters.Subscriber(self, Odometry, '/gt_pose/velodyne')

        ts = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, gt_sub], 20, 0.05)
        ts.registerCallback(self.callback)

        self.get_logger().info("PC2DConverter started (GRID FILTER MODE)")

    def callback(self, pc_msg, gt_msg):

        self.frame_count += 1

        # ===== 1. pose =====
        pos = gt_msg.pose.pose.position
        ori = gt_msg.pose.pose.orientation
        roll, pitch, yaw = t3e.quat2euler([ori.w, ori.x, ori.y, ori.z])

        # ===== 2. 点云 =====
        pts = np.array([
            [p[0], p[1], p[2]]
            for p in point_cloud2.read_points(pc_msg, skip_nans=True)
        ])

        if pts.shape[0] == 0:
            return

        # ===== 3. 转 world =====
        R = self.rpy_to_matrix(roll, pitch, yaw)
        pts_world = (R @ pts.T).T + np.array([pos.x, pos.y, pos.z])

        # ===== 4. 高度带筛选 =====
        mask = np.abs(pts_world[:, 2] - self.slice_z) < self.z_band
        pts_band = pts_world[mask]

        if pts_band.shape[0] == 0:
            self.get_logger().warn("No points in z band")
            return

        # =====================================================
        # ⭐ 核心：XY网格聚类（找“相同位置”）
        # =====================================================
        grid = defaultdict(list)

        for p in pts_band:
            gx = int(p[0] / self.grid_size)
            gy = int(p[1] / self.grid_size)
            grid[(gx, gy)].append(p)

        filtered_points = []

        for (gx, gy), cell_points in grid.items():
            if len(cell_points) >= self.min_points_per_cell:
                # ⭐ 用中心点代表
                cell_points = np.array(cell_points)
                mean_xy = np.mean(cell_points[:, :2], axis=0)

                # ⭐ 强制 z = slice_z
                filtered_points.append([mean_xy[0], mean_xy[1], self.slice_z])

        if len(filtered_points) == 0:
            self.get_logger().warn("No valid clustered points")
            return

        pts_filtered = np.array(filtered_points)

        # =====================================================
        # ⭐ 转 2D 雷达坐标系（只用 XY + yaw）
        # =====================================================
        dx = pts_filtered[:, 0] - pos.x
        dy = pts_filtered[:, 1] - pos.y

        pts_xy = np.stack([dx, dy], axis=1)

        R_yaw_2d = np.array([
            [math.cos(-yaw), -math.sin(-yaw)],
            [math.sin(-yaw),  math.cos(-yaw)]
        ])

        pts_local_xy = (R_yaw_2d @ pts_xy.T).T

        # ===== 5. LaserScan =====
        angles = np.arctan2(pts_local_xy[:, 1], pts_local_xy[:, 0])
        ranges = np.linalg.norm(pts_local_xy, axis=1)

        scan = LaserScan()
        scan.header = Header()
        scan.header.stamp = pc_msg.header.stamp
        scan.header.frame_id = 'velodyne_2d'

        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = (self.angle_max - self.angle_min) / self.horizontal_beams
        scan.range_min = self.min_range
        scan.range_max = self.max_range
        scan.ranges = [float('inf')] * self.horizontal_beams

        for a, r in zip(angles, ranges):
            idx = int((a - scan.angle_min) / scan.angle_increment)
            if 0 <= idx < self.horizontal_beams and self.min_range <= r <= self.max_range:
                scan.ranges[idx] = min(scan.ranges[idx], r)

        self.laser_pub.publish(scan)

        # ===== 日志 =====
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f"[Frame {self.frame_count}] "
                f"Band:{pts_band.shape[0]} | Clustered:{len(filtered_points)}"
            )

    def rpy_to_matrix(self, roll, pitch, yaw):
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll),  math.cos(roll)]
        ])
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx


def main():
    rclpy.init()
    node = PC2DConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
