#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

class SimpleMapper(Node):
    def __init__(self):
        super().__init__('simple_mapper')

        # 地图参数
        self.resolution = 0.05  # 每个栅格 5cm
        self.width = 400        # 20m x 20m
        self.height = 400
        self.origin_x = -10.0   # 地图中心
        self.origin_y = -10.0
        self.map_data = np.zeros((self.height, self.width), dtype=np.int8)

        # 发布 OccupancyGrid
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # 订阅点云和 GT pose
        self.pc_sub = self.create_subscription(PointCloud2, '/velodyne_points', self.pc_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/gt_pose/velodyne', self.pose_callback, 10)

        self.current_pose = None

    def pose_callback(self, msg):
        self.current_pose = msg

    def pc_callback(self, pc_msg):
        if self.current_pose is None:
            return

        # 获取机器人位姿
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        # 清空地图（可改成累积建图）
        self.map_data.fill(-1)

        # 遍历点云
        for point in pc2.read_points(pc_msg, skip_nans=True):
            px, py, pz = point[:3]
            # 旋转点云到世界坐标
            wx = x + math.cos(yaw)*px - math.sin(yaw)*py
            wy = y + math.sin(yaw)*px + math.cos(yaw)*py

            # 转栅格
            ix = int((wx - self.origin_x) / self.resolution)
            iy = int((wy - self.origin_y) / self.resolution)

            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.map_data[iy, ix] = 100  # 占用

        # 发布 OccupancyGrid
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map'
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = self.map_data.flatten().tolist()
        self.map_pub.publish(grid)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
