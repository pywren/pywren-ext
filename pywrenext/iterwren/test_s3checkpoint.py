import boto3
import numpy as np
from pywrenext.iterwren import s3checkpoint

rand_base = np.random.randint(1000000)

def test_simple():
    
    base_checkpoint = "s3://jonas-pywren-132/foo/{}/baz".format(rand_base)
    
    assert s3checkpoint.find_latest_checkpoint_num(base_checkpoint) is None
    data_10 = {'info' : "checkpoint 10"}
    res_url = s3checkpoint.save_checkpoint(base_checkpoint, 10, 
                                           data_10)
    print("res_url=", res_url)
    x_k = s3checkpoint.find_latest_checkpoint_num(base_checkpoint)
    assert x_k == 10
    res = s3checkpoint.load_checkpoint(base_checkpoint, x_k)
    assert res == data_10

    # save new one and find it
    data_20 = {'info' : "checkpoint 20"}
    res_url = s3checkpoint.save_checkpoint(base_checkpoint, 20, 
                                           data_20)

    x_k = s3checkpoint.find_latest_checkpoint_num(base_checkpoint)
    assert x_k == 20
    res = s3checkpoint.load_checkpoint(base_checkpoint, x_k)
    assert res == data_20

    # save old iter, see if we still find new iter
    data_15 = {'info' : "checkpoint 15"}
    res_url = s3checkpoint.save_checkpoint(base_checkpoint, 15, 
                                           data_20)

    x_k = s3checkpoint.find_latest_checkpoint_num(base_checkpoint)
    assert x_k == 20
    res = s3checkpoint.load_checkpoint(base_checkpoint, x_k)
    assert res == data_20

