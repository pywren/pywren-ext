"""
Simple tests of iterwren
"""
import pywrenext.iterwren


def count_func(k, x_k, arg):
    """
    Start at arg and count up
    """
    if k == 0:
        return arg
    else:
        return x_k + 1

def simple_run(x):
    """
    Simple test of the iterator interface
    """

    with pywren.invokers.DummyInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        iter_futures = IE.map(count_func, 3, range(4))


