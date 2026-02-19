# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
<<<<<<< HEAD
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils.logging import print_log

from mmflow import digit_version
from mmflow.apis import multi_gpu_test, single_gpu_test
from mmflow.core import online_evaluation
from mmflow.datasets import build_dataloader, build_dataset
from mmflow.datasets.utils.flow_io import write_flow, write_flow_kitti
from mmflow.models import build_flow_estimator
from mmflow.utils import get_root_logger, setup_multi_processes
=======
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmflow.utils import register_all_modules
>>>>>>> dev


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFlow test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show prediction results at runtime, available when `--show-dir` '
        'is not specified')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved. ')
    parser.add_argument(
<<<<<<< HEAD
        '--eval', type=str, nargs='+', help='evaluation metrics, e.g., "EPE"')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--show-dir', help='directory where visual flow maps will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
=======
        '--wait-time', type=float, default=2, help='the interval of show (s)')
>>>>>>> dev
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # register all modules in mmflow into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
<<<<<<< HEAD

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
=======
>>>>>>> dev

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        # Turn on visualization
        if args.show_dir:
            visualization_hook['show'] = False
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
        elif args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time

<<<<<<< HEAD
    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
=======
>>>>>>> dev
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

<<<<<<< HEAD
    # set multi-process settings
    setup_multi_processes(cfg)

    # The overall dataloader settings
    loader_cfg = {
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    }
    # The specific training dataloader settings
    test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}

    # build the dataloader
    separate_eval = cfg.data.test.get('separate_eval', False)
    if separate_eval:
        # multi-datasets will be built as themselves.
        dataset = [
            build_dataset(dataset) for dataset in cfg.data.test.datasets
        ]
    else:
        # multi-datasets will be concatenated as one dataset.
        dataset = [build_dataset(cfg.data.test)]
    data_loader = [
        build_dataloader(
            _dataset,
            **test_loader_cfg,
            dist=distributed,
        ) for _dataset in dataset
    ]

    # build the model and load checkpoint
    model = build_flow_estimator(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    rank, _ = get_dist_info()

    for i, i_data_loader in enumerate(data_loader):

        if args.out_dir:

            if not distributed:
                outputs = single_gpu_test(
                    model,
                    i_data_loader,
                    out_dir=args.out_dir,
                    show_dir=args.show_dir)
            else:
                outputs = multi_gpu_test(model, i_data_loader, args.tmpdir,
                                         args.gpu_collect)
                if rank == 0:
                    print(f'\nwriting results to {args.out_dir}')
                    for i, output in enumerate(outputs):
                        if args.sparse_flow:
                            write_flow_kitti(output, f'flow_{i}.png')
                        else:
                            write_flow(output, f'flow_{i}.flo')

        if args.eval:
            dataset_name = dataset[i].__class__.__name__
            if hasattr(dataset[i], 'pass_style'):
                dataset_name += f' {dataset[i].pass_style}'
            print_log(
                f'In {dataset_name} '
                f'{online_evaluation(model, i_data_loader, metric=args.eval)}'
                '\n',
                logger=get_root_logger())
=======
    return cfg
>>>>>>> dev


if __name__ == '__main__':
    main()
