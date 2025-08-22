from setuptools import find_packages, setup

setup(
    name='maq-sim2real',
    version='0.0.1',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='MAQ: sim2real for multi-agent quadrupeds',
    url="https://github.com/Nike353/maq-sim2real",  # Update this with your actual repository URL
    python_requires=">=3.8",
    install_requires=[
        "hydra-core>=1.2.0",
        "numpy==1.23.5",
        "rich",
        "ipdb",
        "matplotlib",
        "termcolor",
        "wandb",
        "plotly",
        "tqdm",
        "loguru",
        "meshcat",
        "pynput",
        "scipy",
        "tensorboard",
        "onnx",
        "onnxruntime",
        "opencv-python",
        "joblib",
        "easydict",
        "lxml",
        "numpy-stl",
        'gymnasium>=1.0.0',
    ]
)