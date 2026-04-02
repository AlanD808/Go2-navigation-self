# Go2-navigation-self
It's a small project based on unitree go2 robotic dog and champ controller, aiming at accomplishing mapping and navigating without Nav2. The "3D pointcloud to 2D laserscan" converter has been created to solving the issue which 3D lader would rotate with base_link, causing considerable pointcloud noises. Through transformation of coordinate, 3D pointcloud would first convert to world coordinate, and than select target points convert back to laser coordinate, resulting perfect clear 2D laserscan. The demo mainly achieves that the random creation of 3 waypoints and self-navigation to them sequentially.

The more detailed README is in this online document(only read).
https://kcnebivtbe1b.feishu.cn/wiki/WuJWwuYiYiwj6KkBzFdc4f7jnQb?from=from_copylink
