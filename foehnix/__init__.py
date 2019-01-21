# API to access the main class as foehnix.Foehnix(...) after import foehnix
from foehnix.foehnix import Foehnix
from foehnix.foehnix_filter import foehnix_filter
from foehnix.families import Family, GaussianFamily
from foehnix.iwls_logit import iwls_logit

# __version__
from foehnix import version
__version__ = version.short_version
del version
