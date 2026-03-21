========
Tutorial
========

This tutorial assumes that you have downloaded the code and the Netcdf validation files needed to run the tutorial and example calculations.

The directory structure is assumed to be:: 

  skytropd
  | -- skytropd              <-- Run the tutorial from inside this directory
  |    | -- tutorial.py
  |    | -- functions.py
  |    | -- __init__.py   
  |    | -- metrics.py
  |    | -- TropD_Example_Calculations.py
  | -- ValidationData
  |    | -- *.nc             <-- Collection of netcdf data files used by 
  |    `                         tutorial.py and TropD_Example_Calculations.py
  ` -- ValidationMetrics
       | -- *.nc             <-- Collection of netcdf validation data files with 
       `                         precomputed metrics used by TropD_Example_Calculations.py

.. currentmodule:: skytropd

First import skytropd and some data.

.. code-block:: python

  In [1]: import skytropd as pyt

  In [2]: from skytropd.tutorial import lat, lev, V 

  In [3]: print(V)

.. currentmodule:: skytropd

V is a numpy array containing the mean meridional velocity on (lat, levs). We can calculate the metric of tropical width from the mass streamfunction (PSI) as follows:

.. code-block:: python
  
  In [4]: Phi_sh, Phi_nh = pyt.TropD_Metric_PSI(V[j,:,:], lat, lev)

  In [5]: print(Phi_sh, Phi_nh)

If you already have a precomputed mass streamfunction field, you can calculate the HC edge directly from Psi without recomputing it from meridional wind:

.. code-block:: python

  In [6]: Psi = pyt.TropD_Calculate_StreamFunction(V[j,:,:], lat, lev)

  In [7]: Phi_sh, Phi_nh = pyt.TropD_Metric_PSI(Psi, lat, lev, field_type="PSI")

  In [8]: print(Phi_sh, Phi_nh)

More detailed code examples can be found in the file ``TropD_Example_Calculations.py``.
