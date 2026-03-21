============
Installation
============

Packages
============

You'll need to install the following packages:

* `Python 3.8+ <http://python.org/download/>`_
* `Numpy (>=1.19) and Scipy <https://www.scipy.org/install.html>`_
* `Matplotlib <https://matplotlib.org/users/installing.html>`_
* SkyTropD source code
            
Pip install
===========

  ``pip install skytropd``

Note that this method will only install the package. The additional Netcdf validation files used by the tutorial and example calculations should be obtained from your SkyTropD source tree.


From source control
===================

1. Clone your SkyTropD repository

  ``git clone <your-fork-url>``

2. Compile and install the source
    
  ``cd skytropd``

  ``python setup.py build``

  ``python setup.py install``

From source archive
===================

1. Download the tarball from `here <https://pypi.org/project/skytropd/#files>`_

2. Untar the files 

  ``tar -xzvf skytropd-<version>.tar.gz``
  
  ``cd skytropd-<version>``

3. Compile and install the source
    
  ``cd skytropd``

  ``python setup.py build``

  ``python setup.py install``
