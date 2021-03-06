import struct
import binascii

from pywrenext import iterwren
import pickle
import hashlib
import zlib
import boto3
import copy
from pywrenext.iterwren import s3checkpoint

def crc32_hex_pickle(x):
    data_str = pickle.dumps(x, -1)
    data_bin = struct.pack('!I', zlib.crc32(data_str))
    return binascii.hexlify(data_bin)

def dict_key_filter(d, prefix="_"):
    return {k : v for k, v in d.items() if not k.startswith(prefix)}


def checkpoint_wrapper(f, checkpoint_root, max_attempt_iter, 
                       extra_args = None):
    """
    returns a function conforming to the iterator interface. 
    Note that args is assumed to be a dict, and is hashed
    to get the unique idea

    extra_args are added to the 
    Note that this assumes that args is a dict and we can add
    extra args to it. 

    """
    def iter_func(k, x_k, arg):
        # get the latest checkpoint
        filtered_arg = dict_key_filter(arg, prefix="_")
        default_checkpoint_name = crc32_hex_pickle(filtered_arg).decode('ascii')

        checkpoint_name = arg.get('checkpoint_name', 
                                  default_checkpoint_name)

        checkpoint_path = s3checkpoint.create_checkpoint_path(checkpoint_root, 
                                                              checkpoint_name)

        checkpoint_k = s3checkpoint.find_latest_checkpoint_num(checkpoint_path)
        if checkpoint_k is None:
            checkpoint_k = -1
            checkpoint_x_k = None
        else:
            checkpoint_x_k = s3checkpoint.load_checkpoint(checkpoint_path, 
                                                          checkpoint_k)
        

        # custom args 
        arg = copy.deepcopy(arg)
        if extra_args is not None:
            arg.update(extra_args)
        arg['_checkpoint_path'] = checkpoint_path
        
        if checkpoint_k > max_attempt_iter:
            raise iterwren.IterationDone()
        else:
            k_next = checkpoint_k + 1
            x_next = f(k_next, checkpoint_x_k, arg)
            return s3checkpoint.save_checkpoint(checkpoint_path, k_next, x_next)
    return iter_func

def checkpoint_map(f, total_iter, args, checkpoint_names):
    """
    Simple wrapper around itermap that 
    does the checkpointing
    """
    

"""
I want checkpointing to be TOTALLY TRANSPARENT
except that sometimes I want to MNANUALLY CONTRO THE NAMING


we could sha1-hash the pickled string of the args? 
we could just use the map key

"""



#checkpoint_wrapper(func, max_iter, 
