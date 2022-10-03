Install muon
============

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

Stable version
--------------

``muon`` can be installed `from PyPI <https://pypi.org/project/muon>`_ with ``pip``:
::
	pip install muon


Development version
-------------------

To use a pre-release version of ``muon``, install it from `from the GitHub repository <https://github.com/gtca/muon>`_:
::
	pip install git+https://github.com/gtca/muon


Troubleshooting
---------------

Please see details on installing ``scanpy`` and its dependencies `here <https://scanpy.readthedocs.io/en/stable/installation.html>`_. If there are issues that have not beed described, addressed, or documented, please consider `opening an issue <https://github.com/gtca/muon/issues>`_.

If you encounter the error ``Illegal instruction: 4`` when installing ``muon`` on Apple Silicon (e.g. M1 or M2 chips), try `these steps <https://developer.apple.com/metal/tensorflow-plugin/>`_ that were suggested for a similar error when installing TensorFlow.

Hacking on muon
---------------
For hacking on the package, it is most convenient to do a so-called development-mode install, which symlinks files in your Python package directory to your muon working directory, such that you do not need to reinstall after every change. We use `flit <https://flit.readthedocs.io/en/latest/index.html>`_ as our build system. After installing flit, you can run ``flit install -s`` from within the muon project directory to perform a development-mode install. Happy hacking!
