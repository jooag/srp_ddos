# DDoS detection using Machine Learning
This project intends to detect **DDoS attacks** with **machine learning** techniques. Currently, **Stream Random Patches** with **Hoeffding Trees** are being explored.

Dependencies are:

> river==0.15.0

> numpy==1.24.2

> pandas==1.5.3

## Next steps

- Change plotting: calculate statistics in all data used in training so far
- Change shuffling: it's interesting to keep data in its real order. It may worsen the results, but it's closer to the real word situation
- Try AdaCost: to be able to assign misclassification costs may improve results.
- Try no ensemble: are the ensembles doing anything?
- Run new models on older datasets. It's importante to have data for comparison.
- Try new base estimators: other trees, SVM's (?), NN's (?), etc.
- Try to consume dataset by windows: it may be useful to extract features of windows (by time or number of packets). It's faster and may be more accurate.

> Next meeting at: 05/04/2023

> Next presentation at: 26/04/2023