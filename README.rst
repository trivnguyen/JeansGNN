
==================================================
JeansGNN: Neural Simulation-based Inference with GNN for Jeans Modeling
==================================================

JeansGNN is a neural simulation-based inference framework for Jeans modeling based on Nguyen et al. (2023) [1]_. You can also find our paper on arXiv at `https://arxiv.org/abs/2208.12825`.

JeansGNN can also perform the unbinned Jeans analysis as described in Chang et al. (2021) [2]_.

The framework is built on top of the `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ and `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ library.


Installation
------------

To install JeansGNN, simply run:

.. code-block:: bash

    python setup.py install

Usage
-----
Currently, only the GNN method refered in Nguyen et al. (2023) [1]_ is implemented.
You can find the example of how to use the GNN method in `tutorials/example_training.ipynb`.

The unbinned Jeans analysis in Chang et al. (2021) [2]_ is being migrated to JeansGNN. The original repo can be found at `https://github.com/trivnguyen/dsphs_jeans`.

The rest of the tutorials are under construction.

Documentation
-------------

Under construction.

Contributing
------------

We welcome contributions to JeansGNN! To contribute, please contact Tri Nguyen (tnguy@mit.edu).

License
-------

JeansGNN is licensed under the MIT license. See `LICENSE.md` for more information.

References
----------
.. [1] Tri Nguyen, Siddharth Mishra-Sharma, Reuel Williams, Lina Necib, "Uncovering dark matter density profiles in dwarf galaxies with graph neural networks", *Physical Review D (PRD)*, vol. 107, no. 4, article no. 043015, Feb. 2023, doi: 10.1103/PhysRevD.107.043015.

.. [2] Laura J Chang, Lina Necib, Dark matter density profiles in dwarf galaxies: linking Jeans modelling systematics and observation, *Monthly Notices of the Royal Astronomical Society*, Volume 507, Issue 4, November 2021, Pages 4715 4733, https://doi.org/10.1093/mnras/stab2440