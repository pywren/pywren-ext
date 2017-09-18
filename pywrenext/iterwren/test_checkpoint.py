import boto3
import numpy as np
from pywrenext.iterwren import s3checkpoint

from pywrenext.iterwren import checkpoint
import pywrenext.iterwren
import pywren

def test_checkpoint():
    
    rand_base = np.random.randint(1000000)

    s3_base = "s3://jonas-pywren-132/foo/{}/baz".format(rand_base)

    def increment(k, x_k, args):
        if k == 0:
            return args['val']
        else:
            return x_k + 1
    
    TOTAL_ITERS = 10
    wrapped_func = checkpoint.checkpoint_wrapper(increment, 
                                                 s3_base, TOTAL_ITERS)
    
    states = []
    for i in range(5):
        res0 = wrapped_func(0, None, {'val' : 10})
        s = s3checkpoint.load_checkpoint_url(res0)
        states.append(s)
    assert states == [10, 11, 12, 13, 14]

    states = []
    for i in range(5):
        res0 = wrapped_func(0, None, {'val' : 10})
        s = s3checkpoint.load_checkpoint_url(res0)
        states.append(s)
    assert states == [15, 16, 17, 18, 19]

def test_checkpoint_iterwren():
    rand_base = np.random.randint(1000000)

    s3_base = "s3://jonas-pywren-132/foo/{}/baz".format(rand_base)

    def increment(k, x_k, args):
        if k == 0:
            return args['val']
        else:
            return x_k + 1
    
    TOTAL_ITERS = 10
    wrapped_func = checkpoint.checkpoint_wrapper(increment, 
                                                 s3_base, TOTAL_ITERS)
    
    
    with pywren.invokers.LocalInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        with pywrenext.iterwren.IterExec(wrenexec) as IE:

            iter_futures = IE.map(wrapped_func, 5, [{'val' : i} for i in range(5)])

            pywrenext.iterwren.wait_exec(IE)

        final_results_urls = [f.result() for f in iter_futures]
        final_results = [s3checkpoint.load_checkpoint_url(u) \
                         for u in final_results_urls]
        print(final_results)

    with pywren.invokers.LocalInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        with pywrenext.iterwren.IterExec(wrenexec) as IE:

            iter_futures = IE.map(wrapped_func, 5, [{'val' : i} for i in range(5)])

            pywrenext.iterwren.wait_exec(IE)

        final_results_urls = [f.result() for f in iter_futures]
        final_results = [s3checkpoint.load_checkpoint_url(u) \
                         for u in final_results_urls]
        print(final_results)


def test_checkpoint_iterwren_stopiter():
    rand_base = np.random.randint(1000000)

    s3_base = "s3://jonas-pywren-132/foo/{}/baz".format(rand_base)

    def increment(k, x_k, args):
        if k == 0:
            return args['val']
        else:
            return x_k + 1
    
    TOTAL_ITERS = 3
    wrapped_func = checkpoint.checkpoint_wrapper(increment, 
                                                 s3_base, TOTAL_ITERS)
    
    
    with pywren.invokers.LocalInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        with pywrenext.iterwren.IterExec(wrenexec) as IE:

            iter_futures = IE.map(wrapped_func, 5, [{'val' : i} for i in range(5)])

            pywrenext.iterwren.wait_exec(IE)


        #final_results_urls = [f.result() for f in iter_futures]
        #final_results = [s3checkpoint.load_checkpoint_url(u) \
        #                 for u in final_results_urls]
        #print(final_results)

