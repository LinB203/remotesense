import os

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from matplotlib import pyplot as plt
import mmcv
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm
config_file = r"D:\林彬\mmsegmentation-master\configs\dnlnet\dnl_r101-d8_512x512_160k_ade20k.py"
checkpoint_file = r"D:\林彬\mmsegmentation-master\tools\dnl_r101_yaogan_5\iter_160000.pth"


model = init_segmentor(config_file, checkpoint_file, device='cuda:1')


img_root = r'..\tools\data\test_jpg/jpg/'
save_mask_root = r"..\tools\data\DNLNetpre/"
if not os.path.exists(save_mask_root):
    os.mkdir(save_mask_root)
img_names = os.listdir(img_root)
# print(img_names)
for img_name in tqdm(img_names):
    # test a single image
    img = img_root + img_name
    result = inference_segmentor(model, img)[0]
    img = Image.fromarray(np.uint8(result))
    img.save(save_mask_root + img_name)