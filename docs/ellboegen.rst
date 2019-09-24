.. currentmodule:: foehnix-python

.. _ellboegen-demo:

Data Set Description
====================

The "Tyrolean" data set provides hourly observations
from two stations, namely "Ellbögen" and "Sattelberg" located in
Tyrol, Austria.

Ellbögen is our target station (valley site) located in the Wipp Valley, a
north-south oriented alpine valley on the northern side of the European Alps.
To the north the Wipp Valley opens into the Inn Valley (close to Innsbruck, the
capitol of Tyrol), to the south the valley ends at a narrow gap in the main
Alpine ridge called Brennerpass (:math:`1370~m`; the pass between Austria and Italy)
flanked by mountains (:math:`>2100~m`).
The Wipp Valley is one of the lowest and most distinct cuts trough the Alpine
mountain range and well known for south foehn (north of the Alps).  Station
Sattelberg serves as crest station and provides observations of the upstream
air mass during south foehn events. The station is located on top of the
mountain to the west of the pass.


Loading the Data Set
====================

The call
:func:`foehnix.get_demodata('tyrol') <foehnix.get_demodata>`
returns the combined
data set for both station (Ellbögen and Sattelberg).
In addition, the potential temperature difference between the two stations
is calculated by reducing the dry air temperature from "Sattelberg"
to the height of "Ellbögen" (dry adiabatic lapse rate of 1K per 100m;
stored on ``diff_t``).
Details can be found on the
:func:`~foehnix.get_demodata`
reference page.

.. ipython:: python
   :suppress:

    import numpy as np
    np.set_printoptions(threshold=10)

.. ipython:: python

    import numpy as np
    import foehnix

    # load the data for Tyrol and show a summary:
    data = foehnix.get_demodata('tyrol')
    data.head()

The data set returned is a regular :class:`pandas.DataFrame` object.
Note that the data set is not strictly regular (contains hourly observations,
but some are missing) and contains quite some missing values (``NaN``).
This is not a problem as the functions and methods will take care of missing
values and inflate the time series object
(regular :math:`\rightarrow` strictly regular).

**Important:** The names of the variables in the Tyrolean data set are the
"*standard names*" on which most functions and methods provided by this package
are based on. To be precise:

* **Valley station:** air temperature ``t``, relative humidity ``rh``,
  wind speed ``ff``, wind direction ``dd``
  (meteorological, degrees :math:`\in [0, 360]`)
* **Crest station:** air temperature ``t_crest``,
  relative humidity ``rh_crest``, wind speed ``ff_crest``,
  wind direction ``dd_crest`` (:math:`\in [0, 360]`)
* **Note:** The crest station syntax is different from the *R foehnix* version,
  where a prefix is used (e.g. *crest_ff*).
* **In addition:** Potential temperature difference ``diff_t`` (calculated by
  :func:`~foehnix.get_demodata`)

... however, all functions have arguments which allow to set custom names
(see "[Demos > Viejas (California, USA)](articles/viejas.html)" or
function references).


Based on prior knowledge we define two "foehn wind sectors"
as follows:

* At **Ellbögen** the observed wind direction (``dd``) needs to be
  along valley within 43 and 223 degrees
  (south-easterly; a 180 degree sector).
* At **Sattelberg** the observed wind direction (``dd_crest``) needs to be
  within 90 and 270 degrees (south wind; 180 degree sector).


Estimate Classification Model
=============================

The most important step is to estimate the :class:`foehnix.Foehnix`
classification model. We use the following model assumptions:

* **Main variable**: ``diff_t`` is used as the main covariate to separate 'foehn'
  from 'no foehn' events (potential temperature difference).
* **Concomitant variable**: ``rh`` and ``ff`` at valley site (relative humidity and
  wind speed).
* **Wind filters**: two filters are defined. ``dd = [43, 223]`` for Ellbögen and
  ``dd_crest = [90, 270]`` for Sattelberg (see above).
* **Option switch:** ``switch=True`` as high ``diff_temp`` indicate stable stratification (no foehn).


Run the model and show a summary

.. ipython:: python

    # Estimate the foehnix classification model
    tyrol_filter = {'dd': [43, 223], 'dd_crest': [90, 270]}
    model = foehnix.Foehnix('diff_t', data, concomitant=['rh', 'ff'], filter_method=tyrol_filter, switch=True)

Model summary
-------------

.. ipython:: python

    model.summary(detailed=True)

The data set contains :math:`N = 113952` observations,
:math:`108452` from
the data set itself (``data``) and :math:`5527` due to inflation used to make the
time series object strictly regular.

Due to missing data :math:`38859` observations are not considered
during model estimation (``dd``, ``dd_crest``, ``diff_t``, ``rh``, or ``ff`` missing),
and :math:`50665` are not included in model estimation as they do not
lie within the defined wind sectors (``tyrol_filter``).
Thus, the
:class:`~foehnix.Foehnix` model is based on a total number of
:math:`24428` observations.


Estimated Coefficients
----------------------

The following parameters are estimated for the two
Gaussian clusters:

* No-foehn cluster: :math:`\mu_1 = 5.8, \sigma_1 = 2.64 (parameter scale)`,
* Foehn cluster: :math:`\mu_2 = 0.86, \sigma_2 = 1.33 (parameter scale)`,
* Concomitant model: positive ``rh`` :math:`-5.3` percent on
  relative humidity and a positive ``ff`` effect of :math:`+141.4`
  percent on wind speed

.. ipython:: python

    model.coef['mu1']
    np.exp(model.coef['logsd1'])
    model.coef['mu2']
    np.exp(model.coef['logsd2'])

In other words: if the relative humidity increases the probability that we observed
foehn decreases, while the probability increases with increasing wind speed.


Graphical Model Assessment
==========================

A :class:`~foehnix.Foehnix` object comes with generic plots for graphical model
assessment.

The following figure shows the 'log-likelihood contribution' of

* the main **component** (left hand side of formula),
* the **concomitant** model (right hand side of formula),
* and the **full** log-likelihood sum which is maximised by
  the optimization algorithm.

The abscissa shows (by default) the logarithm of the iterations during
optimization.

.. ipython:: python

    @savefig loglikecontribution.png
    model.plot('loglikcontribution', log=True)

In addition, the coefficient paths during optimization can be visualized:

.. ipython:: python

    @savefig coef.png
    model.plot('coef', log=True)

The left plot shows the parameters of the two components
(:math:`\mu_1, \log(\sigma_1), \mu_2, \log(\sigma_2)`), the
right one the standardized coefficients of the concomitant model.

Last but not least a histogram with the two clusters is plotted.
``'hist`` creates an empirical density histogram separating "no foehn"
and "foehn" events adding the estimated distribution for these two clusters.

.. ipython:: python

    @savefig hist.png
    model.plot('hist')


Time Series Plot
----------------

.. ipython:: python

    @savefig timeseries.png
    model.plot('timeseries', start='2017-02-01', end='2017-02-11', ndays=11)


Hovmoeller Diagram
------------------

The image function plots a Hovmoeller Diagram to assess foehn freqency

.. ipython:: python

    @savefig hovmoeller1.png
    model.plot('image', deltat=3600, deltad=7)

Customized plot which shows the "foehn frequency" with custom
colors and additional contour lines and custom aggregation period
(10-days, 3-hourly).

.. ipython:: python

    import colorspace

    mycmap = colorspace.sequential_hcl(palette='Blue-Yellow', rev=True).cmap(51)

    @savefig hovmoeller2.png
    model.plot('image', deltat=2*3600, deltad=10, cmap=mycmap,
               contours=True, contour_color='w', contour_levels=np.arange(0, 1, 0.05),
               contour_labels=True)
