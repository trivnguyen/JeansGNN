
from setuptools import setup, find_packages

setup(
    name='jeans_gnn',
    version='0.0.0',
    description='GNN for Jeans modeling',
    author='Tri Nguyen',
    author_email='tnguy@mit.edu',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/trivnguyen/JeansGNN/',
    license='MIT',
    install_requires=[
        # list your package's dependencies here
        'numpy',
        'scipy',
        'astropy',
        'torch_geometric',
        'pytorch_lightning',
        'pyyaml',
        'tensorboard',
        'bilby'
    ],
)
