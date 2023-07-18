# -*- coding: utf-8 -*-
from importlib.metadata import version, PackageNotFoundError


__version__ = "0.0.0" # placeholder for dynamic versioning
try:
    __version__ = version("address_ner")
except PackageNotFoundError:
    # package is not installed
    pass
