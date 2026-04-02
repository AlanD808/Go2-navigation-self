#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

class SimpleMapper(Node):
    def __init__(self):
        super().__init__('simple_mapper')

        # 地图参数：扩大地图保证激光点落入地图
        self.resolution = 0.05  # 5cm
        self.width = 800        # 40m
        self.height = 800       # 40m
        self.origin_x = -20.0
        self.origin_y = -20.0

        # 初始化地图为未知
        self.map_data = -np.ones((self.height, self.width), dtype=np.int8)

        # 发布器
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # TF 广播器
        self.tf_broadcaster = TransformBroadcaster(self)

        # 订阅 LaserScan 和 Odometry
        self.laser_sub = self.create_subscription(LaserScan, '/velodyne_2d', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/gt_pose/velodyne', self.odom_callback, 10)

        self.current_pose = None

        self.get_logger().info("SimpleMapper node initialized. Waiting for Odometry and LaserScan...")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.get_logger().info(f"[Odometry] x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    def laser_callback(self, scan: LaserScan):
        if self.current_pose is None:
            self.get_logger().warn("Waiting for Odometry...")
            return

        # 获取机器人位姿
        x = self.current_pose.pose.pose.position.x
        y = self.current_pose.pose.pose.position.y
        q = self.current_pose.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        self.get_logger().info(f"[LaserScan] Received {len(scan.ranges)} points. Sample ranges: {scan.ranges[:10]}")
        self.get_logger().info(f"[LaserScan] Robot pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

        points_in_map = 0
        points_out_map = 0

        # 遍历 LaserScan
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            px = r * math.cos(angle)
            py = r * math.sin(angle)

            # 转到世界坐标
            wx = x + math.cos(yaw)*px - math.sin(yaw)*py
            wy = y + math.sin(yaw)*px + math.cos(yaw)*py

            # 栅格坐标
            ix = int((wx - self.origin_x)/self.resolution)
            iy = int((wy - self.origin_y)/self.resolution)

            if 0 <= ix < self.width and 0 <= iy < self.height:
                # 占用格子
                self.map_data[iy, ix] = 100
                # 标记沿射线的空闲格子
                self.mark_free_line(x, y, wx, wy)
                points_in_map += 1
            else:
                points_out_map += 1

            # 打印部分点的映射情况
            #if i % 50 == 0:
            #    self.get_logger().info(f"Point {i}: world ({wx:.2f},{wy:.2f}) -> map ({ix},{iy})")

        #self.get_logger().info(f"Points in map: {points_in_map}, out of map: {points_out_map}")

        # 发布地图
        self.publish_map()
        # 发布 map -> odom TF
        self.publish_map_tf()

    def mark_free_line(self, rx, ry, px, py):
        """Bresenham-like line for free cells"""
        x0 = int((rx - self.origin_x)/self.resolution)
        y0 = int((ry - self.origin_y)/self.resolution)
        x1 = int((px - self.origin_x)/self.resolution)
        y1 = int((py - self.origin_y)/self.resolution)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < self.width and 0 <= y < self.height and self.map_data[y, x] == -1:
                    self.map_data[y, x] = 0
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            # 标记终点
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                self.map_data[y1, x1] = 100
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < self.width and 0 <= y < self.height and self.map_data[y, x] == -1:
                    self.map_data[y, x] = 0
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            # 标记终点
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                self.map_data[y1, x1] = 100

    def publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map'
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.orientation.w = 1.0
        grid.info.map_load_time = Time(
            sec=int(self.get_clock().now().seconds_nanoseconds()[0]),
            nanosec=int(self.get_clock().now().seconds_nanoseconds()[1])
        )

        # 转换 numpy -> Python list[int]
        grid.data = self.map_data.flatten().tolist()

        self.map_pub.publish(grid)
        self.get_logger().info("Map published.")

        # 打印地图中心样例
        cx = self.width // 2
        cy = self.height // 2
        sample = self.map_data[cy-5:cy+5, cx-5:cx+5]
        self.get_logger().info(f"map_data center 10x10 sample:\n{sample}")

    def publish_map_tf(self):
        """发布静态 map -> odom TF"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
