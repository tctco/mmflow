# Copyright (c) OpenMMLab. All rights reserved.
import copy
<<<<<<< HEAD
from typing import List, Optional, Sequence, Union
=======
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union
>>>>>>> dev

import numpy as np
import torch
from mmcv.ops import Correlation
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmflow.datasets.transforms import Compose
from mmflow.models import build_flow_estimator
from mmflow.structures import FlowDataSample


def init_model(config: Union[str, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None) -> torch.nn.Module:
    """Initialize a flow estimator from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Default to: None.
        device (str): Represent the device. Default to: 'cuda:0'.
        cfg_options (dict, optional): Options to override some settings in the
            used config. Default to: None.
    Returns:
        nn.Module: The constructed flow estimator.
    """

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    config.model.train_cfg = None
    model = build_flow_estimator(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


<<<<<<< HEAD
def inference_model(
    model: torch.nn.Module,
    img1s: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
    img2s: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
    valids: Optional[Union[str, np.ndarray, Sequence[str],
                           Sequence[np.ndarray]]] = None
) -> Union[List[np.ndarray], np.ndarray]:
=======
def inference_model(model: torch.nn.Module, img1s: Union[str, np.ndarray],
                    img2s: Union[str, np.ndarray]) -> List[FlowDataSample]:
>>>>>>> dev
    """Inference images pairs with the flow estimator.

    Args:
        model (nn.Module): The loaded flow estimator.
        img1s (str/ndarray or sequence[str/ndarray]): Either image files or
            loaded images.
        img2s (str/ndarray or sequence[str/ndarray]): Either image files or
            loaded images.
        valids (str/ndarray or list[str/ndarray], optional): Either mask files
            or loaded mask. If the predicted flow is sparse, valid mask will
            filter the output flow map.
    Returns:
        If img-pairs is a list or tuple, the same length list type results
        will be returned, otherwise return the flow map from image1 to image2
        directly.
        List[FlowDataSample]: the predicted flows from img1s to img2s.
            For a single img-pair, its prediction is in the data field
            of ``pred_flow_fw`` in FlowDataSample.
    """
    if not isinstance(img1s, (list, tuple)):
        img1s = [img1s]
        img2s = [img2s]
<<<<<<< HEAD
        valids = [valids]
        is_batch = False
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
=======
>>>>>>> dev

    cfg = model.cfg
    if isinstance(cfg.test_dataloader, list):
        cfg = copy.deepcopy(cfg.test_dataloader[0].dataset)
    else:
        cfg = copy.deepcopy(cfg.test_dataloader.dataset)

    if isinstance(img1s[0], np.ndarray):
        cfg.pipeline[0].type = 'LoadImageFromWebcam'

    # as load annotation is for online evaluation
    # there is no need to load annotation.
    if dict(type='LoadAnnotations') in cfg.pipeline:
        cfg.pipeline.remove(dict(type='LoadAnnotations'))
    if dict(type='LoadAnnotations', sparse=True) in cfg.pipeline:
        cfg.pipeline.remove(dict(type='LoadAnnotations', sparse=True))

    test_pipeline = Compose(cfg.pipeline)
<<<<<<< HEAD
    datas = []
    valid_masks = []
    for img1, img2, valid in zip(img1s, img2s, valids):
=======
    datas = defaultdict(list)
    for img1, img2 in zip(img1s, img2s):
>>>>>>> dev
        # prepare data
        if isinstance(valid, str):
            # there is no real example to test the function for loading valid
            # masks
            valid = mmcv.imread(valid, flag='grayscale')
        if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
            # directly add img and valid mask
            data = dict(img1=img1, img2=img2, valid=valid)
        else:
            # add information into dict
<<<<<<< HEAD
            data = dict(
                img_info=dict(filename1=img1, filename2=img2),
                img1_prefix=None,
                img2_prefix=None,
                valid=valid)
        data['img_fields'] = ['img1', 'img2']
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
        valid_masks.append(valid)
=======
            data = dict(img1_path=img1, img2_path=img2)
        # build the data pipeline
        data = test_pipeline(data)
        datas['inputs'].append(data['inputs'])
        datas['data_samples'].append(data['data_samples'])
>>>>>>> dev

    datas = model.data_preprocessor(datas, False)
    inputs, data_samples = datas['inputs'], datas['data_samples']

    for m in model.modules():
        assert not isinstance(
            m, Correlation
        ), 'CPU inference with Correlation is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model.predict(inputs, data_samples)

<<<<<<< HEAD
    if valid_masks[0] is not None:
        # filter the output flow map
        for result, valid in zip(results, valid_masks):
            if result.get('flow', None) is not None:
                result['flow'] *= valid
            elif result.get('flow_fw', None) is not None:
                result['flow_fw'] *= valid

    if not is_batch:
        # only can inference flow of forward direction
        if results[0].get('flow', None) is not None:
            return results[0]['flow']
        if results[0].get('flow_fw', None) is not None:
            return results[0]['flow_fw']
    else:
        return results
=======
    return results
>>>>>>> dev
