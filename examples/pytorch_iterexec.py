import numpy as np
import time
import pywren
import logging
import threading
import pywrenext.iterwren
import pickle

import daiquiri
 
daiquiri.setup() # level=logging.INFO)
#dep_log = logging.getLogger('multyvac.dependency-analyzer')
#dep_log.setLevel('DEBUG')
iw = logging.getLogger('pywrenext.iterwren.iterwren')
iw.setLevel('DEBUG')

#local_wrenexec = pywren.local_executor()

# with pywrenext.iterwren.IterExec(local_wrenexec) as IE:

#     def foo(x):
#         return x + 1
#     f = local_wrenexec.map(foo, [1.0])[0]
#     f.result()


def offset_counter(k, x_k, offset):
    time.sleep(60)
    if k == 0:
        return offset
    else:
        return x_k + 1
   
def ser(model, optimizer):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    return {'model_state' :  model_state, 
            'optimizer_state' : optimizer_state}

def pt_iter(k, x_k, args):

    # generic setup
    model = Net()
    opt_lr = args['learning_rate']
    opt_momentum = args.get('opt_momentum', 0.5)
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=opt_lr, 
                          momentum=opt_momentum)

    # just return initial state
    if k == 0:
        return ser(model, optimizer)

    model.load_state_dict(x_k['model_state'])
    if x_k > 1:
        optimizer.load_state_dict(x_k['optimizer_state'])

    # do normal stuff


    return(model_state, optimizer_state)

wrenexec = pywren.local_executor()


with pywrenext.iterwren.IterExec(wrenexec) as IE:
    
    iter_futures = IE.map(offset_counter, 10, range(100), save_iters=True)
    while not IE.alldone():
        print("whee")
        IE.process_pending()
        time.sleep(1)

iter_futures_hist = [f.iter_hist for f in iter_futures]
pickle.dump({'iter_futures_hist' : iter_futures_hist}, 
            open("runlog.pickle", 'wb'), -1)
print("results dumped")
