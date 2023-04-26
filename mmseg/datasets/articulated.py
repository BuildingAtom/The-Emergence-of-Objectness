import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

import os
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger

@DATASETS.register_module()
class ArticulatedDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'foreground')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(ArticulatedDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for _i, line in enumerate(f):
                    arrays = line.strip().split()
                    file_dir = arrays[0]
                    base_dir = os.path.join(img_dir, file_dir)
                    rgb_dir = osp.join(base_dir, 'rgb/')
                    depth_dir = osp.join(base_dir, 'depth/')
                    file_list = arrays[1:]

                    img_info = dict(filename=[rgb_dir, file_list], depth_folder = depth_dir)

                    if ann_dir is not None:
                        #seg_name1 = 'JPEGImages/480p/bear/00080'
                        #seg_name2 = seg_name1
                        #seg_map1 = osp.join(ann_dir, seg_name1 + seg_map_suffix)
                        #seg_map2 = osp.join(ann_dir, seg_name2 + seg_map_suffix)
                        #img_info['ann'] = dict(seg_map=[seg_map1,seg_map2])

                        # if 'JPEGImages' in file_dir and '_all_frames' not in file_dir:
                        #     seg_name = osp.join(file_dir, file_list[0][:-4] + '.png').replace('JPEGImages', 'Annotations')
                        # else:
                        seg_name = osp.join(base_dir, 'mask/', file_list[0])
                        img_info['ann'] = dict(seg_map=seg_name)
                    img_infos.append(img_info)
        else:
            assert 1==0
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_file = osp.join(img_dir, img)
                img_info = dict(filename=img_file)
                if ann_dir is not None:
                    seg_map = osp.join(ann_dir,
                                       img.replace(img_suffix, seg_map_suffix))
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)
        res = self.pipeline(results)
        # print(res)
        return True, res