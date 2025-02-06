from setuptools import setup, find_packages

setup(
    name='calib_proj',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy',
        'matplotlib',
        # 'calib-commons @ git+https://github.com/tflueckiger/calib-commons.git'
    ],
)