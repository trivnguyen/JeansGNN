
==================================================
JeansGNN: Neural Simulation-based Inference with GNN for Jeans Modeling
==================================================

JeansGNN is a neural simulation-based inference framework for Jeans modeling based on Nguyen et al. (2023) [1]_. You can also find our paper on arXiv at `https://arxiv.org/abs/2208.12825`.

JeansGNN can also perform the unbinned Jeans analysis as described in Chang & Necib (2021) [2]_.

The framework is built on top of the `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ and `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ library.

:Authors:
    Tri Nguyen,
    Siddharth Mishra-Sharma,
    Reuel Williams,
    Lina Necib,
:Maintainer:
    Tri Nguyen (tnguy@mit.edu)
:Version: 0.0.0 (2023-04-14)

Installation
------------

To install JeansGNN, simply clone the repo and install with `pip`:

.. code-block:: bash

    git clone https://github.com/trivnguyen/JeansGNN.git
    pip install .

This should install all the dependencies as well. If you want to install the dependencies separately, please see the section below.

Dependencies
------------

The following dependencies are required to run this project:

- Python 3.6 or later
- NumPy 1.22.3 or later
- SciPy 1.9.1 or later
- Astropy 5.2.2 or later
- PyTorch Geometric 2.1.0 or later
- PyTorch Lightning 1.7.6 or later
- PyYAML 5.4.1 or later
- Tensorboard 2.7.0 or later
- Bilby 2.1.0 or later

To install the dependencies separately, you can use `pip`:

.. code-block:: bash

    pip install -r requirements.txt

It is recommended to use a virtual environment to manage the dependencies and avoid version conflicts. You can create a virtual environment and activate it using the following commands:

.. code-block:: bash

    python -m venv env
    source env/bin/activate (Linux/MacOS)
    env\Scripts\activate.bat (Windows)

Once the virtual environment is activated, you can install the dependencies using pip as usual.

Usage
-----
An example of the graph-based simulation-based inference method in Nguyen et al. (2023) [1]_ can be found at ``tutorials/example_training.ipynb``.

An example of the binned Jeans analysis in Chang & Necib (2021) [2]_ can be found at ``tutorials/example_binned_jeans.ipynb``.

The rest of the tutorials are under construction. More to come!

Documentation
-------------

Under construction.

Contributing
------------

We welcome contributions to JeansGNN! To contribute, please contact Tri Nguyen (tnguy@mit.edu).

License
-------

JeansGNN is licensed under the MIT license. See ``LICENSE.md`` for more information.

References
----------
.. [1] Tri Nguyen, Siddharth Mishra-Sharma, Reuel Williams, Lina Necib, "Uncovering dark matter density profiles in dwarf galaxies with graph neural networks", *Physical Review D (PRD)*, vol. 107, no. 4, article no. 043015, Feb. 2023, https://doi.org/10.1103/PhysRevD.107.043015

.. [2] Laura J Chang, Lina Necib, Dark matter density profiles in dwarf galaxies: linking Jeans modelling systematics and observation, *Monthly Notices of the Royal Astronomical Society*, Volume 507, Issue 4, November 2021, Pages 4715 4733, https://doi.org/10.1093/mnras/stab2440
