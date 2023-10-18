from setuptools import setup, find_packages

setup(

    name='pitchdetector',

    version='0.0.1',

    author='Cyberblasta',

    author_email='cyberblasta@gmail.com',

    packages=find_packages(),

    install_requires=[

        'numpy',
        'onnxruntime',
        'PyQt5'

    ],

)
