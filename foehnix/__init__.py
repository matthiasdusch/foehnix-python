# API to access the main class as foehnix.Foehnix(...) after import foehnix
from .foehnix import Foehnix
from .foehnix_filter import foehnix_filter, filter_summary
from .families import Family, GaussianFamily, LogisticFamily
from .iwls_logit import iwls_logit
from .demodata import get_demodata

# __version__
from foehnix import version
__version__ = version.short_version
del version
