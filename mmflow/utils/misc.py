# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD
import glob
import os.path as osp
import warnings


def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    It will be used when automatically resume, modified from
    https://github.com/open-mmlab/mmdetection/blob/dev-v2.20.0/mmdet/utils/misc.py.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.

    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """ # noqa
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        # `count` is iteration number, as checkpoints are saved as
        # 'iter_xx.pth' or 'epoch_xx.pth' and xx is iteration number.
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path
=======
import numpy as np
import torch
import torch.distributed as dist
from mmengine.dist import get_dist_info


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed. All workers must call
    this function, otherwise it will deadlock. This method is generally used in
    `DistributedSampler`, because the seed should be identical across all
    processes in the distributed group.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()
>>>>>>> dev
