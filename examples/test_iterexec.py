import numpy as np
import time
import pywren
import logging
import threading
import pywrenext.iterwren




import daiquiri
 
daiquiri.setup(level=logging.INFO)


def offset_counter(k, x_k, offset):
    if k == 0:
        return offset
    else:
        return x_k + 1
    

with pywrenext.iterwren.IterExec(local_wrenexec) as IE:
    
    iter_futures = IE.map(offset_counter, 10, [4], save_iters=True)
    #pywrenext.iterwren.wait_exec(IE)
    while not IE.alldone():
        print("whee")
        IE.process_pending()
        time.sleep(1)
