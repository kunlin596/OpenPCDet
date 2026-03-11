import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .custom.custom_dataset import CustomDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'CustomDataset': CustomDataset,
}

# Deferred dataset backends — imported on demand in build_dataloader().
# Avoids mandatory deps (av2, nuscenes-devkit, etc.) when only using KITTI.
_DEFERRED_DATASETS = {
    'NuScenesDataset': ('.nuscenes.nuscenes_dataset', 'NuScenesDataset'),
    'WaymoDataset': ('.waymo.waymo_dataset', 'WaymoDataset'),
    'PandasetDataset': ('.pandaset.pandaset_dataset', 'PandasetDataset'),
    'LyftDataset': ('.lyft.lyft_dataset', 'LyftDataset'),
    'ONCEDataset': ('.once.once_dataset', 'ONCEDataset'),
    'Argo2Dataset': ('.argo2.argo2_dataset', 'Argo2Dataset'),
}


def _resolve_dataset(name):
    """Resolve a dataset class by name, importing deferred backends on demand."""
    if name in __all__:
        return __all__[name]
    if name in _DEFERRED_DATASETS:
        import importlib
        module_path, cls_name = _DEFERRED_DATASETS[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, cls_name)
    raise KeyError(f"Unknown dataset: {name!r}")


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = _resolve_dataset(dataset_cfg.DATASET)(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
