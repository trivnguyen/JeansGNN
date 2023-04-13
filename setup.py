
from setuptools import setup, find_packages

setup(
    name='jeans_gnn',
    version='0.0.0',
    description='GNN for Jeans modeling',
    author='Tri Nguyen',
    author_email='tnguy@mit.edu',
    packages=find_packages(),
    install_requires=[
        # list your package's dependencies here
        'numpy',
        'pandas',
        'scipy',
        'astropy'
    ],
    # entry_points={
    #     'console_scripts': [
    #         # list any command-line scripts here
    #     ],
    # },
)
