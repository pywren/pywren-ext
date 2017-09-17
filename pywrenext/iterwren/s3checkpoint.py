import fnmatch
import pickle
import boto3
import os
import pywren.wrenutil
from io import BytesIO
from pywren.serialize.cloudpickle import cloudpickle

CHECKPOINT_DIGITS = 8
CHECKPOINT_FORMAT = "{{:0{:d}d}}".format(CHECKPOINT_DIGITS)

## S3-specific

def create_checkpoint_path(checkpoint_root, checkpoint_name):
    return os.path.join(checkpoint_root, checkpoint_name)

def find_latest_checkpoint_num(s3_url):
    """
    Look for /foo/%08d under the path
    """
    s3_bucket, s3_key = pywren.wrenutil.split_s3_url(s3_url)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket)

    filenames = []
    for obj in bucket.objects.filter(Prefix=s3_key):
        if fnmatch.fnmatch(obj.key, '*' + '?' * CHECKPOINT_DIGITS):
            dig_key = obj.key[-CHECKPOINT_DIGITS:]

            if dig_key.isdigit():
                filenames.append((int(dig_key), obj.key))

    filenames = sorted(filenames, key=lambda x : x[0])

    if len(filenames) == 0:
        return None
    latest_iteration = filenames[-1][0]
    return latest_iteration

def create_checkpoint_path_iter(checkpoint_base, k):
    return os.path.join(checkpoint_base, CHECKPOINT_FORMAT.format(k))

def load_checkpoint(checkpoint_base, k):
    """
    return unpickled object 
    """

    s3_url = create_checkpoint_path_iter(checkpoint_base, k)

    return load_checkpoint_url(s3_url)

def load_checkpoint_url(s3_url):
    s3 = boto3.resource('s3')
    s3_bucket, s3_key = pywren.wrenutil.split_s3_url(s3_url)
    bucket = s3.Bucket(s3_bucket)
    # FIXME switch to streaming? Does this double our mem
    bytes_fid = BytesIO()
    bucket.download_fileobj(s3_key, bytes_fid)
    bytes_fid.seek(0)
    return pickle.load(bytes_fid)

    

def save_checkpoint(checkpoint_base, k, obj):
    """
    Save the checkpoint for iteration k, return the URL
    
    """
    

    s3_url = create_checkpoint_path_iter(checkpoint_base, k)

    s3 = boto3.resource('s3')
    s3_bucket, s3_key = pywren.wrenutil.split_s3_url(s3_url)
    bucket = s3.Bucket(s3_bucket)
    bucket.put_object(Key=s3_key, Body = cloudpickle.dumps(obj))

    return s3_url
