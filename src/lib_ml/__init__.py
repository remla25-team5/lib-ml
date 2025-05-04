from .preprocessing import preprocess_dataset, preprocess_element

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0" 


__all__ = ['__version__', 'preprocess_dataset', 'preprocess_element']
