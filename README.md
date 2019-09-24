# foehnix - Python version


[![Build Status](https://travis-ci.com/matthiasdusch/foehnix-python.svg?branch=master)](https://travis-ci.com/matthiasdusch/foehnix-python)
[![codecov](https://codecov.io/gh/matthiasdusch/foehnix-python/branch/master/graph/badge.svg)](https://codecov.io/gh/matthiasdusch/foehnix-python)
[![Repository Status](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/badges/latest/wip.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# A Toolbox for Automated Foehn Classification based on Mixture Models

_foehnix_ package provides a toolbox for automated probabilistic foehn
wind classification based on two-component mixture models (**foehn**
m**ix**ture models).
_foehnix_ models are a special case of the general flexible mixture model
class ([Fraley 2002](#fraley2000), [Leisch 2004](#leisch2004), [Gr&uuml;n
2007](#gruen2007), [Gr&uuml;n 2008](#gruen2008)), an unsupervised statistical
model to identify unobserveable clusters or components in data sets.

The application of mixture models for an automated classification of foehn winds
has first been proposed by
[Plavcan et al. (2014)](#plavcan2014).
The "Community Foehn Classification Experiment"
shows that the method performs similar compared to another semi-automatic classification,
foehn experts, students, and weather enthusiasts (see [Mayr 2019](mayr2018)).

Aim of this software package:

* provide **easy-to-use functions** for classification
* create **probabilistic foehn classification**
* **easy scalability** (can be applied to large data sets)
* **reproducibility** of the results
* create results which are **comparable to other locations**


### Important Links

* [_foehnix Python_ documentation](https://matthiasdusch.github.io/foehnix-python)
* [R version of foehnix](https://github.com/retostauffer/Rfoehnix), also available on github.
* [_R foehnix_ documentation](http://retostauffer.github.io/Rfoehnix), currently more comprehensive than the Python documentation.


# Installation
The package is not yet published via the _Python Package Index_
([PyPi](https://pypi.org)) but will be made available as soon as finished.

Currently the easiest way to install _foehnix Python_ on Linux is via github and pip:  

``` bash
git clone https://github.com/matthiasdusch/foehnix-python
cd foehnix-python
pip install -e .
```

## Create classification

Once the observation data have been imported, one can start doing the
classification. The _foehnix_ package comes with two demo data sets,
one for Southern California (USA) and one for Tyrol (A).
The documentation provides a walk-through on how to start using
_foehnix_:

* Demo for [Ellbögen (Tyrol, A)](https://retostauffer.github.io/Rfoehnix/articles/ellboegen.html)
* Demo for [Viejas (California, USA)](https://retostauffer.github.io/Rfoehnix/articles/viejas.html)

### References

<p id="mayr2018">
Mayr GJ, Plavcan D, Laurence A, Elvidge A, Grisogono B, Horvath K, Jackson P,
Neururer A, Seibert P, Steenburgh JW, Stiperski I, Sturman A, Ve&#269;enaj
&#381;, Vergeiner J, Vosper S, Z&auml;ngl G (2018).  The Community Foehn
Classification Experiment.
<i>Bulletin of the American Meteorological Society</i>, <b>99</b>(11), 2229&mdash;2235,
<a href="https://doi.org/10.1175/BAMS-D-17-0200.1" target="_blank">10.1175/BAMS-D-17-0200.1</a>
</p>

<p id="plavcan2014">
Plavcan D, Mayr GJ, Zeileis A (2014).
Automatic and Probabilistic Foehn Diagnosis with a Statistical Mixture Model.
<i>Journal of Applied Meteorology and Climatology</i>, <b>53</b>(3), 652&mdash;659,
<a href="https://dx.doi.org/10.1175/JAMC-D-13-0267.1" target="_blank">10.1175/JAMC-D-13-0267.1</a>
</p>

<p id="hastie2009">
Hastie T, Tibshirani R, Friedman J (2009).
Fitting Logistic Regression Models. In <i>The Elements of Statistical Learning</i>
(Chapter 4.4.1), 2<i>nd</i> edition, ISBN 978-0387848570.
<a href="https://web.stanford.edu/~hastie/ElemStatLearn/" target="_blank">PDF download</a>
</p>

<p id="gruen2008">
Gr&uuml;n B, Friedrich L (2008).
FlexMix Version 2: Finite Mixtures with Concomitant Variables and Varying and Constant Parameters.
<i>Journal of Statistical Software, Articles</i>, <b>28</b>(4), 1&mdash;35,
doi:<a href="https://dx.doi.org/10.18637/jss.v028.i04" target="_blank">10.18637/jss.v028.i04</a>
</p>

<p id="gruen2007">
Gr&uuml;n B, Leisch F (2007).
Fitting Finite Mixtures of Generalized Linear Regressions in _R_.
<i>Computational Statistics & Data Analysis</i>, <b>51</b>(11),
doi:<a href="https://dx.doi.org/10.1016/j.csda.2006.08.014" target="_blank">10.1016/j.csda.2006.08.014</a>
</p>

<p id="leisch2004">
Friedrich L (2004).
FlexMix: A General Framework for Finite Mixture Models and Latent Class Regression in <i>R</i>.
<i>Journal of Statistical Software, Articles</i>, <b>11</b>(8), 1&mdash;18,
doi:<a href="https://dx.doi.org/10.18637/jss.v011.i08" target="_blank">10.18637/jss.v011.i08</a>
</p>

<p id="fraley2000">
Fraley C, Raftery AE (2000).
Model-Based Clustering, Discriminant Analysis, and Density Estimation.
<i>Journal of the American Statistical Association</i>, <b>97</b>(458), 611&mdash;631,
doi:<a href="https://dx.doi.org/10.1198/016214502760047131" target="_blank">10.1198/016214502760047131</a>
</p>

<p id="mccullagh1999">
McCullagh P, Nelder JA (1999).
Likelihood functions for binary data. In <i>Generalized Linear Models</i> (Chapter 4.4),
2<i>nd</i> edition, ISBN 0-412-31760-5.
</p>
