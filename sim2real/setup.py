from setuptools import setup, find_packages

setup(
    name="sim2real",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mujoco",
        "pyyaml",
        "scipy",
        "onnxruntime",
        "pynput",
        "ipdb",
        "termcolor",
        "sshkeyboard",
        "vuer[all]",
        "loguru",
        "loop_rate_limiters",
        "pygame",
        "matplotlib",
        "zmq",
        "meshcat"
    ]
)