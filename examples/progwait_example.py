import pywren
import time
import numpy as np

from pywrenext.progwait import progwait

wrenexec = pywren.default_executor()

def sleep(x):
    time.sleep(x)
    return x


futures = wrenexec.map(sleep, np.arange(10) + 10)
futures_done, _ = progwait(futures)
print(pywren.get_all_results(futures_done))

