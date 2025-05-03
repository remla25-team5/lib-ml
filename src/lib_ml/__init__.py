from .placeholder import test_package

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0" 


__all__ = ['__version__', 'test_package']