"""
Wait with a progress bar

"""

import tqdm
import pywren

def progwait(fs, desc="", notebook=False):
    """
    A tqdm-based progress bar that looks nice and gives you an
    idea of how long until your futures are done
    """

    N = len(fs)
    result_count = 0
    fs_dones = []
    fs_notdones = fs

    if notebook:
        import tqdm.tqdm_notebook
        tqdm_func = tqdm.tqdm_notebook
    else:
        from tqdm import tqdm
        tqdm_func = tqdm

    with tqdm_func(total=N, desc=desc) as pbar:
        while len(fs_dones) < N:
            new_fs_dones, new_fs_notdones = pywren.wait(fs_notdones,
                                                        return_when=pywren.ANY_COMPLETED)
            fs_dones += new_fs_dones
            fs_notdones = new_fs_notdones
            pbar.update(len(new_fs_dones))
    return fs_dones, fs_notdones

