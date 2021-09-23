from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='pra_utils', 
    version='0.0.1',  
    description='A package which adds extra features to pyroomacoustics', 
    long_description=long_description,  
    long_description_content_type='text/markdown', 

    url='https://github.com/enceladus2000/pra_utils',
    author='Tanmay Bhonsale', 
    author_email='f20180903@goa.bits-pilani.ac.in',

    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='acoustics, pyroomacoustics',  # Optional
    package_dir={'': 'src'},  # Optional

    packages=find_packages(where='src'),  # Required

    python_requires='>=3.6, <4',

    install_requires=[
        "numpy",
        "numpy_stl>=2.16.0",
        "matplotlib>=3",
        "pyroomacoustics==0.4.3",
        "dataclasses",
        "PyYAML~=5.4.1",
    ],
)
