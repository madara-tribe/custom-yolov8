__version__ = '8.0.122'

from custom.yolo.engine.model import YOLO
from custom.yolo.utils.checks import check_yolo as checks
#from custom.yolo.utils.downloads import download

__all__ = '__version__', 'YOLO', 'checks', 'download'  # allow simpler import
