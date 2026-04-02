#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from PIL import Image
import yaml
import os
import numpy as np  # 添加 numpy 导入

class MapSaver(Node):
    def __init__(self):
        super().__init__('map_saver')

        # 订阅 /map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # 保存路径为当前工作目录
        self.save_directory = os.getcwd()
        self.get_logger().info(f"Map will be saved to {self.save_directory}")

    def map_callback(self, msg: OccupancyGrid):
        """当接收到地图数据时，保存为 .pgm 和 .yaml 文件"""
        # 获取地图数据
        map_data = msg.data
        resolution = msg.info.resolution
        origin = msg.info.origin.position
        width = msg.info.width
        height = msg.info.height

        # 转换地图数据为 numpy 数组
        import numpy as np
        map_array = np.array(map_data).reshape((height, width))

        # 保存 .pgm 文件
        self.save_pgm(map_array, resolution, origin.x, origin.y)

        # 保存 .yaml 文件
        self.save_yaml(resolution, origin.x, origin.y)

        # 停止节点
        self.get_logger().info("Map saved successfully. Shutting down...")
        rclpy.shutdown()

    def save_pgm(self, map_data, resolution, origin_x, origin_y):
        """将地图数据保存为 .pgm 文件"""
        # 转换为图像格式
        img = np.zeros_like(map_data, dtype=np.uint8)
        img[map_data == -1] = 205  # 未知区域为灰色
        img[map_data == 0] = 254   # 空闲区域为白色
        img[map_data == 100] = 0   # 占用区域为黑色

        # 保存为 PGM 文件
        pgm_filename = os.path.join(self.save_directory, 'map.pgm')
        im = Image.fromarray(img)
        im.save(pgm_filename)
        self.get_logger().info(f"Map saved as {pgm_filename}")

    def save_yaml(self, resolution, origin_x, origin_y):
        """将地图元数据保存为 .yaml 文件"""
        yaml_filename = os.path.join(self.save_directory, 'map.yaml')
        yaml_dict = {
            'image': 'map.pgm',
            'resolution': resolution,
            'origin': [origin_x, origin_y, 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196
        }
        with open(yaml_filename, 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)
        self.get_logger().info(f"Map metadata saved as {yaml_filename}")

def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
