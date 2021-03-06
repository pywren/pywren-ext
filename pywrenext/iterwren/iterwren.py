#import threading
import logging
from six.moves import queue
import numpy as np
import pywren
import time
import enum

class IterationDone(Exception):
    """
    Raise this for variable-run iterations when done. 
    Conceptually similar to built-in StopIteration but:
    1. wanted something distinct so we have more control
    2. wanted a name that signaled this was "successful" done

    """

    pass

class IterFutureState(enum.Enum):
    new = 0
    running = 1
    success = 1
    error = 2

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

    def result(self):
        return self.current_future.result()


def get_map_pos(futures):
    return [l.original_map_pos for l in futures]


class IterExec(object):
    """
    The IterExec has a map() simnilar to regular pywren which 
    returns a list of iter futures. 

    each invocation of `map` creates a map_id which tracks
    the associated futures
    """


    def __init__(self, wrenexec):
        self.wrenexec = wrenexec

        self.next_map_id = 0
        self.active_iters = {}
        self.wrapped_funcs = {}  # Hack

        self._process_pending_count = 0

    def __enter__(self):
        logger.debug("entered IterExec")
        #self.thread = threading.Thread(target=self._run_loop)
        #self.thread.start()
        return self

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
        pywren_futures = self.wrenexec.map(self.wrapped_funcs[current_map_id], wrapped_args, 
                                           exclude_modules=EXCLUDE_MODULES)
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
        active_map_ids = list(self.active_iters.keys())
        logger.info("ppc={} begin process pending len_active_map_ids={}".format(self._process_pending_count, len(active_map_ids)))

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

            ## call WAIT on everyone 
            logger.debug("map_id={} starting status check".format(map_id))
            for cs, flist in f_by_callset_id.items():
                logger.debug("map_id={} calling wait for callset_id={} len_futures={}".format(map_id,cs, len(flist)))

                # this will trigger an update on all of them
                fs_done, fs_notdone = pywren.wait(flist, return_when=pywren.ALWAYS, # ANY_COMPLETED, 
                                                  WAIT_DUR_SEC=1)
                logger.debug("map_id={} wait done for callset_id={} len_fs_done={}".format(map_id,cs, len(fs_done)))

            logger.debug("map_id={} status check done for all f in map_id".format(map_id))

            to_advance = []
            to_remove = []
            still_waiting = []
            for f in iter_futures:
                pwf = f.current_future
                if f.current_iter == f.max_iter:
                    to_remove.append(f)
                else:
                    if pwf.succeeded():
                        to_advance.append(f)
                    elif pwf.errored():
                        logger.debug("map_id={} map_pos={} errored on iter {}".format(map_id, 
                                                                                      f.original_map_pos, 
                                                                                      f.current_iter))
                        to_remove.append(f)
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

                wrapped_func = self.wrapped_funcs[map_id]
                logger.debug("map_id={} invoking new map with {} args".format(map_id, len(wrapped_args)))
                pywren_futures = self.wrenexec.map(wrapped_func,
                                                   wrapped_args,
                                                   exclude_modules=EXCLUDE_MODULES)
                logger.debug("map_id={} invoking new map done".format(map_id))
                for f, pwf in zip(to_advance, pywren_futures):
                    if f.save_iters:
                        f.iter_hist.append(f.current_future)
                    f.current_future = pwf
                    f.current_iter += 1


            # remove these from current map id
            to_remove_map_pos = [f.original_map_pos for f in to_remove]
            self.active_iters[map_id] = [f for f in self.active_iters[map_id] if f.original_map_pos \
                                         not in to_remove_map_pos]
            if len(self.active_iters[map_id]) == 0:
                logger.debug("map_id={} deleted".format(map_id))
                del self.active_iters[map_id]
                del self.wrapped_funcs[map_id]
        logger.info("ppc={} end process pending".format(self._process_pending_count))
        self._process_pending_count += 1

    def alldone(self):
        return len(self.active_iters) == 0

    def __exit__(self, *args):
        print("ended IterExec")
        #self.stop = True
        #self.thread.join()

IE_WAIT_SLEEP = 1.0

def wait_exec(IE, callback=None):
    wait_num = 0 
    while not IE.alldone():
        IE.process_pending()
        time.sleep(IE_WAIT_SLEEP)
        wait_num += 1
        if callback is not None:
            callback(wait_num)
