
import numpy as np
import pywren


class IterRunner(object):

    def __init__(self, wrenexec):
        self.wrenexec = wrenexec

    def __enter__(self):
        print("entered IterRunner")
        return self

    def __exit__(self, *args):
        print("ended iterrunner")
        pass
