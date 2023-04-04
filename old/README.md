# Streaming Random Patches for detecting DDoS

Code based on [this repository](https://gitlab.com/yaissa/srp_ddos).
Dataset is [Bot-IOT](https://research.unsw.edu.au/projects/bot-iot-dataset).
Raw CSV files should be extracted in dataset subfolder.
Classes were balanced using [ADASYN](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.ADASYN.html).
Numpy has to be at version 1.23 because of [scikit-multiflow incompatibility](https://github.com/scikit-multiflow/scikit-multiflow/issues/312).