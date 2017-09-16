import numpy as np
import time
import pywren
import logging
import threading
import pywrenext.iterwren
import pickle




import daiquiri

def constant_t():
 
    daiquiri.setup() # level=logging.INFO)
    #dep_log = logging.getLogger('multyvac.dependency-analyzer')
    #dep_log.setLevel('DEBUG')
    iw = logging.getLogger('pywrenext.iterwren.iterwren')
    iw.setLevel('DEBUG')

    def offset_counter(k, x_k, offset):
        time.sleep(60)
        if k == 0:
            return offset
        else:
            return x_k + 1

    wrenexec = pywren.default_executor()

    with pywrenext.iterwren.IterExec(wrenexec) as IE:

        iter_futures = IE.map(offset_counter, 10, range(100), save_iters=True)
        pywrenext.iterwren.wait_exec(IE)

    iter_futures_hist = [f.iter_hist for f in iter_futures]
    pickle.dump({'iter_futures_hist' : iter_futures_hist}, 
                open("benchmark_futures_data.pickle", 'wb'), -1)
    print("results dumped")

def random_delay():

    daiquiri.setup() # level=logging.INFO)
    #dep_log = logging.getLogger('multyvac.dependency-analyzer')
    #dep_log.setLevel('DEBUG')
    iw = logging.getLogger('pywrenext.iterwren.iterwren')
    iw.setLevel('DEBUG')
    t1 = time.time()

    def offset_counter(k, x_k, offset):
        time.sleep(np.random.randint(10, 40))
        if k == 0:
            return offset
        else:
            return x_k + 1

    wrenexec = pywren.default_executor()

    TOTAL_ITER = 2
    with pywrenext.iterwren.IterExec(wrenexec) as IE:

        iter_futures = IE.map(offset_counter, TOTAL_ITER, range(100), 
                              save_iters=True)
        pywrenext.iterwren.wait_exec(IE)

    iter_futures_hist = [f.iter_hist for f in iter_futures]
    t2 = time.time()
    pickle.dump({'iter_futures_hist' : iter_futures_hist, 
                 'TOTAL_ITER' : TOTAL_ITER, 
                 'time' : t2-t1}, 
                open("benchmark_futures_data.{}.random.pickle".format(TOTAL_ITER), 'wb'), -1)
    print("results dumped")
    


random_delay()
