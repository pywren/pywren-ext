"""
Simple tests of iterwren
"""
import pywren
import pywren.invokers
import pywrenext.iterwren


def count_func(k, x_k, arg):
    """
    Start at arg and count up
    """
    if k == 0:
        return arg
    else:
        return x_k + 1

def except_func(k, x_k, arg):
    """
    throw an exception when x_k == arg
    """
    fid = open("/tmp/log.txt", 'a')
    fid.write("{} {} {} \n".format(k, x_k, arg))
    fid.close()
    if k == arg:
        raise Exception("k == arg")

    if k == 0:
        return 0
    else:
        return x_k + 1


def test_simple_run():
    """
    Simple test of the iterator interface
    """

    with pywren.invokers.LocalInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        with pywrenext.iterwren.IterExec(wrenexec) as IE:

            iter_futures = IE.map(count_func, 3, range(4))

            pywrenext.iterwren.wait_exec(IE)
            final_results = [f.result() for f in iter_futures]
            assert final_results == [3, 4, 5, 6]


def test_invoke_error():
    """
    invoke an actual error
    """

    with pywren.invokers.LocalInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        with pywrenext.iterwren.IterExec(wrenexec) as IE:

            iter_futures = IE.map(except_func, 10, [2])
            print("mapped")
            pywrenext.iterwren.wait_exec(IE)
            assert iter_futures[0].current_iter == 2


def test_invoke_error_map():
    """
    invoke an actual error
    """

    with pywren.invokers.LocalInvoker("/tmp/task") as iv:

        wrenexec = pywren.local_executor(iv)

        with pywrenext.iterwren.IterExec(wrenexec) as IE:

            iter_futures = IE.map(except_func, 10, [12, 3, 5, 20])
            print("mapped")
            pywrenext.iterwren.wait_exec(IE)
            all_final_iters = [f.current_iter for f in iter_futures]
            print(all_final_iters)
            assert all_final_iters == [10, 3, 5, 10]
