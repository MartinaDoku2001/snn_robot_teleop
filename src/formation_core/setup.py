from setuptools import setup

package_name = 'formation_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/configs', [
            'configs/default.yaml', 'configs/eval_suite.yaml', 'configs/stress_suite.yaml']),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    extras_require={'plots': ['matplotlib'], 'test': ['pytest']},
    zip_safe=True,
    maintainer='Martina Doku',
    maintainer_email='martina.doku.t7@dc.tohoku.ac.jp',
    description='Task definition and evaluation harness for networked leader-follower control.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'formation-run = formation_core.cli:main_run',
            'formation-suite = formation_core.cli:main_suite',
            'formation-sweep = formation_core.sweep:main',
            'formation-figures = formation_core.figures:main',
        ],
    },
)
