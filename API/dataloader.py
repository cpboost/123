from .dataloader_DRS import load_data as load_DRS
from .dataloader_DRS_mask import load_data as load_DRS_mask
from .dataloader_kth import load_data as load_kth
from .dataloader_kth_mask import load_data as load_kth_mask
def load_data(rate,dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'DRS':
        return load_DRS(rate, batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'DRS_mask':
        return load_DRS_mask(rate, batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'kth':
        return load_kth(rate, batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'kth_mask':
        return load_kth_mask(rate, batch_size, val_batch_size, data_root, num_workers)