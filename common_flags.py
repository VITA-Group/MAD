from argparse import Namespace
import os

root_dir = os.path.join(os.path.expanduser('~'), 'MAD')
hdd_dir = '/valida/path/on/your/machine/MAD'
ImageNet_dir = '/valida/path/on/your/machine/imagenet_pytorch_download'

json_dir = os.path.join(root_dir, 'class_info')
label_path = os.path.join(root_dir, 'dataset', 'labels.txt')
dataset_dir = os.path.join(hdd_dir, 'dataset')
data_download_dir = os.path.join(hdd_dir, 'download')
prediction_dir = os.path.join(root_dir, 'predictions')
compare_dir = os.path.join(root_dir, 'compare_results')
select_dir = os.path.join(root_dir, 'selected_50')
plot_dir = os.path.join('plot_result')

COMMON_FLAGS = Namespace()
COMMON_FLAGS.root_dir = root_dir
COMMON_FLAGS.hdd_dir = hdd_dir
COMMON_FLAGS.json_dir = json_dir
COMMON_FLAGS.label_path = label_path
COMMON_FLAGS.dataset_dir = dataset_dir
COMMON_FLAGS.data_download_dir = data_download_dir
COMMON_FLAGS.prediction_dir = prediction_dir
COMMON_FLAGS.compare_dir = compare_dir
COMMON_FLAGS.select_dir = select_dir
COMMON_FLAGS.plot_dir = plot_dir
COMMON_FLAGS.ImageNet_dir = ImageNet_dir
