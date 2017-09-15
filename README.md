# PyWren-Ext : Helper functions and useful utilities for PyWren

This is a collection of useful utilities not yet ready
for inclusion to mainline pywren or segregation into their own
packages. 


## For Developers
If you add a utility or chunk of code please do not
import from the top-level package. We expect a lot
of different requirements for these projects, ranginging
from matplotlib and seaborn to more esoteric packages. It's 
better for someone to get an import error upon trying to import it. 

### Organization:
If you're just creating something simple (such as `progwait.py`)
feel free to keep it at the top level. Otherwise create
a subdirectory. 

* `tests/myext` for tests (pytest/unittest)
* `examples/myext` for examples (including ipython notebooks and assoc code)

   
