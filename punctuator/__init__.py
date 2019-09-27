VERSION = (0, 9, 2)
__version__ = '.'.join(map(str, VERSION))
try:
    from punc import Punctuator
except ImportError:
    pass
