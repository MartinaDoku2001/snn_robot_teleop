from setuptools import setup

package_name = 'formation_rl'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Martina Doku',
    maintainer_email='martina.doku.t7@dc.tohoku.ac.jp',
    description='PPO controller for the centralized formation task (contract v2.0).',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'formation-rl-train = formation_rl.cli:main_train',
            'formation-rl-benchmark = formation_rl.cli:main_benchmark',
            'formation-rl-figures = formation_rl.figures:main',
        ],
    },
)
