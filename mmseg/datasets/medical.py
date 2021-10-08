import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MedicalDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('others', 'obj')

    PALETTE = [[242,234,218], [69,185,124]]
    # CLASSES = ('others', 'water')
    #
    # PALETTE = [[242, 234, 218], [51,163,220]]

    def __init__(self, split, **kwargs):
        super(MedicalDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
