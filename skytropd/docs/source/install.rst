============
Installation
============

Packages
============

You'll need to install the following packages:

* `Python 3.8+ <http://python.org/download/>`_
* `Numpy (>=1.19) and Scipy <https://www.scipy.org/install.html>`_
* `Matplotlib <https://matplotlib.org/users/installing.html>`_
* `SkyTropD source code <https://github.com/QianyeSu/Skytropd>`_
            
Pip install
===========

  ``pip install skytropd``

Note that this method will only install the package. The additional Netcdf validation files used by the tutorial and example calculations should be obtained from your SkyTropD source tree.


From source control
===================

1. Clone your SkyTropD repository

  ``git clone https://github.com/QianyeSu/Skytropd.git``

2. Compile and install the source
    
  ``cd skytropd``

  ``python setup.py build``

  ``python setup.py install``


