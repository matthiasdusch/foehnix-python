.. foehnix-python documentation master file, created by
   sphinx-quickstart on Tue Jan  8 14:42:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 1
   :caption: globaltoc
   :hidden:

   api.rst
   foehnix-demo.rst
   simulation.rst


Toolbox for Automated Foehn Classification based on Mixture Models
******************************************************************


The **foehnix** package provides a toolbox for automated probabilistic foehn
wind classification based on two-component mixture models (**foehn**
m\ **ix**\ ture models). This method has first been proposed by
:cite:`plavcan_automatic_2013` and compared to
another semi-automatic classification, foehn experts, students, and weather
enthusiasts in the “Community Foehn Classification Experiment”
:cite:`mayr_community_2018`.

Foehn mixture models are a special case of the general flexible mixture model
class (:cite:`fraley_modelbased_2002`, :cite:`leisch_flexmix_2004`,
:cite:`grun_fitting_2007`, :cite:`grun_flexmix_2008`),
an unsupervised statistical model to identify unobserveable clusters or
components in data sets. **foehnix** allows to estimate two-component mixture
models with additional concomitants.

Some of the features:

- Gaussian or logistic components with optional censoring or truncation.
- Concomitant variables for the probability model.
- Model assessment based on graphical output and information criteria.
- Automatic handling of missing values in the data set.

.. figure:: savefig/timeseries.png
    :width: 100%




References
==========

.. bibliography:: references.bib
   :style: unsrt