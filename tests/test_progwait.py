import pywren
import time
from pywrenext.progwait import progwait

wrenexec = pywren.default_executor()

def sleep(x):
    time.sleep(x)
    return x


futures = wrenexec.map(sleep, [10]*10)
progwait(futures)

