#!/usr/bin/env python3
# pc2d_converter.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import message_filters
import numpy as np
import transforms3d.euler as t3e
import math
from tf2_ros import StaticTransformBroadcaster  # --- TF STATIC ADDED ---

class PC2DConverter(Node):
    def __init__(self):
        super().__init__('pc2d_converter')

        # ----------------- Parameters -----------------
        self.horizontal_beams = 440  # Velodyne horizontal
        self.vertical_beams = 16     # Velodyne vertical
        self.min_range = 0.05
        self.max_range = 130.0
        self.angle_min = -math.pi
        self.angle_max = math.pi

        # ----------------- Publishers -----------------
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
        # --- END TF STATIC ADDED ---

        # ----------------- Subscribers -----------------
        pc_sub = message_filters.Subscriber(self, PointCloud2, '/velodyne_points')
        gt_sub = message_filters.Subscriber(self, Odometry, '/gt_pose/velodyne')

        # Approximate time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, gt_sub],
            queue_size=20,
            slop=0.05  # 10ms tolerance
        )
        ts.registerCallback(self.synced_callback)
        self.get_logger().info("Subscribed and time-synchronizer initialized.")

    def synced_callback(self, pc_msg: PointCloud2, gt_msg: Odometry):
        self.get_logger().info(f"Processing PointCloud frame at time {pc_msg.header.stamp.sec}.{pc_msg.header.stamp.nanosec}")

        # 1. 获取 GT pose
        x = gt_msg.pose.pose.position.x
        y = gt_msg.pose.pose.position.y
        z = gt_msg.pose.pose.position.z
        qx = gt_msg.pose.pose.orientation.x
        qy = gt_msg.pose.pose.orientation.y
        qz = gt_msg.pose.pose.orientation.z
        qw = gt_msg.pose.pose.orientation.w

        # 2. 转换四元数为 roll, pitch, yaw
        roll, pitch, yaw = t3e.quat2euler([qw, qx, qy, qz])  # 注意 transforms3d 的顺序 [w, x, y, z]

        # 3. 读取点云为 numpy array
        points_list = []
        for p in point_cloud2.read_points(pc_msg, skip_nans=True):
            points_list.append([p[0], p[1], p[2]])
        points = np.array(points_list)
        if points.shape[0] == 0:
            self.get_logger().warn("Empty point cloud, skipping...")
            return

        # ----------------- 地面点过滤（新增） -----------------
        z_threshold = 0.4
        points = points[np.abs(points[:,2]) > z_threshold]

        if points.shape[0] == 0:
            self.get_logger().warn("All points filtered out (ground removed).")
            return
        # --------------------------------------------------

        # 4. 构建旋转矩阵，只去除 roll, pitch（保持 yaw）
        # 旋转点云到水平坐标系
        R_roll = np.array([[1,0,0],
                           [0, math.cos(-roll), -math.sin(-roll)],
                           [0, math.sin(-roll), math.cos(-roll)]])
        R_pitch = np.array([[math.cos(-pitch),0,math.sin(-pitch)],
                            [0,1,0],
                            [-math.sin(-pitch),0,math.cos(-pitch)]])
        R = R_pitch @ R_roll  # 先roll再pitch
        points_h = (R @ points.T).T

        # 5. 平移至 world 坐标（GT pose XY位置）
        points_world = points_h #+ np.array([x, y, z])

        # 6. 投影到 XY 平面，生成 LaserScan 数据
        angles = np.arctan2(points_world[:,1], points_world[:,0])
        ranges = np.linalg.norm(points_world[:,0:2], axis=1)

        # 创建 LaserScan 消息
        scan_msg = LaserScan()
        scan_msg.header = Header()
        scan_msg.header.stamp = pc_msg.header.stamp
        scan_msg.header.frame_id = 'velodyne_2d'
        scan_msg.angle_min = self.angle_min
        scan_msg.angle_max = self.angle_max
        scan_msg.angle_increment = (self.angle_max - self.angle_min)/self.horizontal_beams
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = self.min_range
        scan_msg.range_max = self.max_range
        scan_msg.ranges = [float('inf')]*self.horizontal_beams  # 初始化为 inf

        # 将点云映射到 LaserScan 的索引
        for a, r in zip(angles, ranges):
            idx = int((a - scan_msg.angle_min)/scan_msg.angle_increment)
            if 0 <= idx < self.horizontal_beams and self.min_range <= r <= self.max_range:
                # 保留最近的点
                scan_msg.ranges[idx] = min(scan_msg.ranges[idx], r)

        # 发布
        self.laser_pub.publish(scan_msg)
        self.get_logger().info(f"LaserScan published with {np.count_nonzero(np.isfinite(scan_msg.ranges))} valid points.")

def main(args=None):
    rclpy.init(args=args)
    node = PC2DConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
