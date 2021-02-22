# Projet-PSA
This repository is dedicated to PSA project. It is a private repository so just people with given access can scripts and data analysis. It contains for the moment two scripts for reading and denoising the signals using wavelets transformations. The next step is to extract some features related to time-frequency from physiological signals in order to perform potential clustering in the futur.


The first one `read_physio.py` reads physiological data in a compact way from directory for all subjects and construct dataframes for each one of them.
The second script `denoising.py` contains necessary functions to denoise physiological signals using wavelets. Two methods are coded `VisuShrink` and `SureShrink`.
