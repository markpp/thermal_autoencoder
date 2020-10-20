# README #

This repo lets you train a basic variational autoencoder.

### Overview ###

* extract_datasets.py extracts crops from a specific view point.
* train.py trains and tests a model using the extracted crops.
* embed.py lets load a model and output embeddings of the extracted crops.
* plot_distribution.py

### extract_datasets ###

* Choose view point "view" variable and determine whether the other options are correct.
* Set flow=False if you want just the [intensity] otherwise crops are stored containing [intensity, flow_x, flow_y].

### train ###

* train mode, no surprises.

### embed ###

* produces reconstructions, reconstruction losses, and latent representation and stores it as .npy.

### plot_distribution ###

* Loads numpy arrays from the .npy file, fits PCA to latent representation vectors, and visualizes the distribution in 2D using the two most significant components.
