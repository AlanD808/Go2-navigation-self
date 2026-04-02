from setuptools import find_packages, setup

package_name = 'pc2_to_2d'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','numpy','transforms3d'],
    zip_safe=True,
    maintainer='alan',
    maintainer_email='alan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pc2d_converter = pc2_to_2d.pc2d_converter:main',  # 注册节点
        ],
    },
)
