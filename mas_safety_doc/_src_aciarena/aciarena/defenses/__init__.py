try:
    from .safety_filter import SafetyFilter
except ModuleNotFoundError:
    SafetyFilter = None
from .aci_sentinel import ACISentinel


def BertDetector(*args, **kwargs):
    from .bert_detector import BertDetector as _BertDetector

    return _BertDetector(*args, **kwargs)
