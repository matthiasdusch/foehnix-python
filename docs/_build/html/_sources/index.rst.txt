.. foehnix-python documentation master file, created by
   sphinx-quickstart on Tue Jan  8 14:42:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 1
   :caption: globaltoc
   :hidden:

   api.rst
   ellboegen.rst
   simulation.rst



foehnix - Python version
========================


.. image:: https://travis-ci.com/matthiasdusch/foehnix-python.svg?branch=master
   :target: https://travis-ci.com/matthiasdusch/foehnix-python

.. image:: https://codecov.io/gh/matthiasdusch/foehnix-python/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/matthiasdusch/foehnix-python

.. image:: https://www.repostatus.org/badges/latest/wip.svg

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg

.. image:: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
   :height: 60
   :target: https://github.com/matthiasdusch/foehnix-python

A Toolbox for Automated Foehn Classification based on Mixture Models
--------------------------------------------------------------------

*foehnix* package provides a toolbox for automated probabilistic foehn
wind classification based on two-component mixture models (**foehn**
m**ix**ture models).
*foehnix* models are a special case of the general flexible mixture model
class  (:cite:`fraley_modelbased_2002`, :cite:`leisch_flexmix_2004`,
:cite:`grun_fitting_2007`, :cite:`grun_flexmix_2008`),
an unsupervised statistical
model to identify unobserveable clusters or components in data sets.

The application of mixture models for an automated classification of foehn winds
has first been proposed by
:cite:`plavcan_automatic_2013`.
The "Community Foehn Classification Experiment"
shows that the method performs similar compared to another semi-automatic classification,
foehn experts, students, and weather enthusiasts (see :cite:`mayr_community_2018`)

Aim of this software package:

* provide **easy-to-use functions** for classification
* create **probabilistic foehn classification**
* **easy scalability** (can be applied to large data sets)
* **reproducibility** of the results
* create results which are **comparable to other locations**


Important Links
~~~~~~~~~~~~~~~

* `foehnix Python on Github <https://github.com/matthiasdusch/foehnix-python>`_
* `R version of foehnix <https://github.com/retostauffer/Rfoehnix>`_, also available on github.
* `R foehnix documentation <http://retostauffer.github.io/Rfoehnix>`_, currently more comprehensive than the Python documentation.


Installation
~~~~~~~~~~~~
The package is not yet published via the *Python Package Index*
(`PyPi <https://pypi.org>`_) but will be made available as soon as finished.

Currently the easiest way to install *foehnix Python* on Linux is via github and pip:

.. code-block:: console

   git clone https://github.com/matthiasdusch/foehnix-python
   cd foehnix-python
   pip install -e .


Create classification
---------------------

Once the observation data have been imported, one can start doing the
classification. The *foehnix* package comes with two demo data sets,
one for Southern California (USA) and one for Tyrol (A).
The documentation provides a walk-through on how to start using
*foehnix*:

* Demo for :ref:`Ellbögen (Tyrol, A) <ellboegen-demo>`
* Demo for `Viejas (California, USA) <https://retostauffer.github.io/Rfoehnix/articles/viejas.html>`_


.. figure:: savefig/timeseries.png
    :width: 100%



References
==========

.. bibliography:: references.bib
   :style: unsrt