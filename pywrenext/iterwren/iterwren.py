#import threading
import logging
from six.moves import queue
import numpy as np
import pywren
import time


EXCLUDE_MODULES=['tqdm']

logger = logging.getLogger(__name__)

def iter_wrapper(func):
    def f(a):
        
        k, x_k, arg = a  # unpack a
        return func(k, x_k, arg)
    return f

class IterFuture(object):
    def __init__(self, func, parent_ie, 
                current_future, arg, 
                current_iter, max_iter, map_pos, save_iters):
        self.func = func
        self.parent_ie = parent_ie # is this necessary? 
        self.current_future = current_future
        self.arg = arg
        self.current_iter = current_iter
        self.max_iter = max_iter
        self.original_map_pos = map_pos
        self.save_iters = save_iters
        self.iter_hist = []

def get_map_pos(futures):
    return [l.original_map_pos for l in futures]

        
class IterExec(object):
    

    def __init__(self, wrenexec):
        self.wrenexec = wrenexec
        #self.thread = None
        #self.queue = queue.Queue()
        #self.stop = False
        
        self.next_map_id = 0
        self.active_iters = {}
        self.wrapped_funcs = {}  # Hack
        
    def __enter__(self):
        logger.debug("entered IterExec")
        #self.thread = threading.Thread(target=self._run_loop)
        #self.thread.start()
        return self
    
    def _run_loop(self):
        BLOCK_SEC_MAX = 2
        while not self.stop:
            try:
                res = self.queue.get(True, BLOCK_SEC_MAX)
                func, iters, args = res
                print("got a thing", str(func))
                
                f = self.wrenexec.call_async(iter_wrapper, (func, 0, args))
                self.queue.task_done()

            except queue.Empty:
                pass
        
    def call_async(self, func, iters, arg):
        return self.map(func, iters, [arg])
    
    def map(self, func, iters, args, save_iters=False):
        # FIXME check func's attributes and... stuff
        # FIXME prudence suggests we try serializing here so we can throw a sane error
        
        current_map_id = self.next_map_id
        logger.debug("initial map map_id={}".format(current_map_id))

        wrapped_args = [(0, None, a) for a in args]
        
        self.wrapped_funcs[current_map_id] = iter_wrapper(func)
        logger.debug("map_id={} invoking initial map with {} args".format(current_map_id, len(wrapped_args)))
        pywren_futures = self.wrenexec.map(self.wrapped_funcs[current_map_id], wrapped_args, exclude_modules=EXCLUDE_MODULES)
        logger.debug("map_id={} invocation done".format(current_map_id))
        iter_futures = []
        for a_i, (a, f) in enumerate(zip(args, pywren_futures)):
            
            iter_future = IterFuture(self, func, f, a, 
                                     0, iters, map_pos = a_i, 
                                     save_iters=save_iters)
            iter_futures.append(iter_future)
            
        self.active_iters[current_map_id] = iter_futures
        
        self.next_map_id += 1
        logger.debug("map returning {} futures".format(len(iter_futures)))
        return iter_futures
    
    def process_pending(self):
        logger.debug("process pending")
        active_map_ids = list(self.active_iters.keys())
        for map_id in active_map_ids:
            logger.debug("processing map_id={} {}".format(map_id, len(self.active_iters[map_id])))
            # get a future set
            iter_futures = self.active_iters[map_id]
            # group these by callset ID 
            f_by_callset_id = {}
            for f in iter_futures:
                pwf = f.current_future
                if pwf.callset_id not in f_by_callset_id:
                    f_by_callset_id[pwf.callset_id] = []
                f_by_callset_id[pwf.callset_id].append(pwf)
            for cs, flist in f_by_callset_id.items():
                logger.debug("map_id={} calling wait for callset_id = {}".format(map_id,cs))

                # this will trigger an update on all of them
                fs_done, fs_notdone = pywren.wait(flist, return_when=pywren.ANY_COMPLETED)
            
            to_advance = []
            to_remove = []
            still_waiting = []
            
            for f in iter_futures:
                pwf = f.current_future
                if f.current_iter == f.max_iter:
                    to_remove.append(f)
                else:
                    if pwf.done(): # fixme handle error! 
                        to_advance.append(f)
                    else:
                        still_waiting.append(f)
            logger.debug("map_id={} to_advance={}".format(map_id, get_map_pos(to_advance)))
            logger.debug("map_id={} to_remove={}".format(map_id, get_map_pos(to_remove)))
            logger.debug("map_id={} still_waiting={}".format(map_id, get_map_pos(still_waiting)))
            
            if len(to_advance) > 0:

                # construct next invocation
                wrapped_args = [(f.current_iter + 1, 
                                 f.current_future.result(),
                                 f.arg) for f in to_advance ]
                # FIXME don't take this from func 0 
                wrapped_func = self.wrapped_funcs[map_id]
                logger.debug("map_id={} invoking new map with {} args".format(map_id, len(wrapped_args)))
                pywren_futures = self.wrenexec.map(wrapped_func, 
                                                   wrapped_args, 
                                                   exclude_modules=EXCLUDE_MODULES)
                logger.debug("map_id{} done with new invoke".format(map_id))
                for f, pwf in zip(to_advance, pywren_futures):
                    if f.save_iters:
                        f.iter_hist.append(f.current_future)
                    f.current_future = pwf
                    f.current_iter += 1

                new_map_id = self.next_map_id 

                self.active_iters[new_map_id] = to_advance
                self.wrapped_funcs[new_map_id] = wrapped_func
                self.next_map_id += 1
            
            # remove these from current map id
            original_map_pos_filter = [f.original_map_pos for f in (to_advance + to_remove)]
            self.active_iters[map_id] = [f for f in self.active_iters[map_id] if f.original_map_pos \
                                         not in original_map_pos_filter]
            if len(self.active_iters[map_id]) == 0:
                logger.debug("deleting map_id={}".format(map_id))
                del self.active_iters[map_id]

    def alldone(self):
        return len(self.active_iters) == 0    

    def __exit__(self, *args):
        print("ended IterExec")
        #self.stop = True
        #self.thread.join()

IE_WAIT_SLEEP = 1.0

def wait_exec(IE):
    while not IE.alldone():
        IE.process_pending()
        time.sleep(IE_WAIT_SLEEP)
