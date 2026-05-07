from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HMR2 as a package',
    name='hmr2',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'torch',
        'torchvision',
        'pytorch-lightning',
        'smplx==0.1.28',
        'pyrender',
        'opencv-python',
        'yacs',
        'scikit-image',
        'einops',
        'timm',
        'webdataset',
        'dill',
        'pandas',
        # Removed chumpy git link to prevent build triggers
    ],
    extras_require={
        'all': [
            # Removed detectron2 git link completely
        ],
    },
)