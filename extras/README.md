Some extra scripts that are outside the scope of this package. These include:
- `samples_galaxies.py` to sample the 6D kinematics of galaxies from the potential and distribution function
  - The sampling process is done by the `Agama` library (http://agama.software/), requires a lot of extra dependencies
  - Example config files are stored in `configs/galaxy_models`
- `preprocess_galaxies.py` to preprocess the galaxies (projection + error convolve) into 2D positions and line-of-sight velocities for GNN
  - Example config files are stored in `configs/dataset_models`

In the future, these scripts might be converted into another packages if demand is high.
