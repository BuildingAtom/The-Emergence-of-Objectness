import os.path as osp
import os
import random

import pdb
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 load_num=2,
                 step_limit=1,
                 is_train=True,
                 use_gauss_blur=False,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.use_gauss_blur = use_gauss_blur
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.is_train=is_train
        self.load_num = load_num
        self.step_limit = step_limit

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        depth_folder = None
        if results['img_info'].get('depth_folder') is not None:
            depth_folder = results['img_info']['depth_folder']

        folder, fn_list = filename

        if self.is_train: 
            st=random.randint(0, len(fn_list)-1); num=self.load_num; step=random.randint(1, self.step_limit)
            #st=0; num=2; step=1
            pass
        else:
            st=0; num=1; step=1

        if len(fn_list[st:]) < step * num:
            # print('[frame num] less than step*num len=', len(fn_list), 'st=', st)
            while len(fn_list[st:]) < step*num :
                fn_list = fn_list + fn_list[::-1]
        fn_list = [fn_list[it] for it in range(st, st+num*step, step)]
        imgs=[]
        depths=[]
        for fn in fn_list:
            fp_fn = os.path.join(folder, fn)
            img_bytes = self.file_client.get(fp_fn)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if depth_folder is not None:
                fp_fn_depth = os.path.join(depth_folder, fn)
                depth_bytes = self.file_client.get(fp_fn_depth)
                depth = mmcv.imfrombytes(depth_bytes, flag="unchanged", backend="pillow")
                depths.append(depth)
            if self.to_float32:
                img = img.astype(np.float32)
            if self.use_gauss_blur:
                import scipy.ndimage as ndimage
                img = ndimage.gaussian_filter(img, sigma=(3, 3, 0), order=0)

            if img.shape[0] > img.shape[1]:
                img = np.transpose(img, [1,0,2])
            imgs.append(img)
        #imgs = np.asarray(imgs)

        #img_bytes = self.file_client.get(filename)
        #img = mmcv.imfrombytes(
        #    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        #if self.to_float32:
        #    img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = imgs
        results['depth'] = depths
        results['img_shape'] = imgs[0].shape
        results['ori_shape'] = imgs[0].shape
        # Set initial values for default meta_keys
        results['pad_shape'] = imgs[0].shape
        results['scale_factor'] = 1.0
        num_channels = 3
        if len(depths) == len(imgs):
            results['ori_img'] = results['img']
            results['img'] = [np.concatenate((i.astype(np.float32),d[...,None].astype(np.float32)),axis=-1,dtype=np.float32) for i,d in zip(imgs, depths)]
            results['img_shape'] = imgs[0].shape
            results['pad_shape'] = imgs[0].shape
            num_channels = 4
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename1 = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename1 = results['ann_info']['seg_map']
        
        img_bytes = self.file_client.get(filename1)
        flow_x = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        
        if flow_x.shape[0] > flow_x.shape[1]:
            flow_x = np.transpose(flow_x, [1,0,2])

        #flow_x_bin = bin_it(flow_x)
        flow_x_bin = flow_x
        flow_y_bin = flow_x_bin

        results['flow_x'] = flow_x_bin
        results['flow_y'] = flow_y_bin
        results['seg_fields'].append('flow_x')
        results['seg_fields'].append('flow_y')
        return results 

        img_bytes = self.file_client.get(filename1)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
