import struct
import binascii

from pywrenext import iterwren
import pickle
import hashlib
import zlib
import boto3

from pywrenext.iterwren import s3checkpoint

def crc32_hex_pickle(x):
    data_str = pickle.dumps(x, -1)
    data_bin = struct.pack('!I', zlib.crc32(data_str))
    return binascii.hexlify(data_bin)

def checkpoint_wrapper(f, checkpoint_root, max_attempt_iter):
    """
    returns a function conforming to the iterator interface
    """
    def iter_func(k, x_k, arg):
        # get the latest checkpoint
        
        default_checkpoint_name = crc32_hex_pickle(arg).decode('ascii')

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
