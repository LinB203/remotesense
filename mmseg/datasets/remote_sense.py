import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RometeSenseDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('others', 'forest', 'road', 'building', 'water')

    PALETTE = [[242,234,218], [69,185,124], [143,75,46], [116,120,124], [51,163,220]]
    # CLASSES = ('others', 'water')
    #
    # PALETTE = [[242, 234, 218], [51,163,220]]

    def __init__(self, split, **kwargs):
        super(RometeSenseDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
