import pywren
import time
import cPickle as pickle
import boto3

def sleep_test(x):
    time.sleep(x)
    return x + 1

if __name__ == "__main__":


    N = 1024
    SLEEP_DUR = 100


    wrenexec = pywren.default_executor()
    futures = wrenexec.map(sleep_test, [SLEEP_DUR] * N)

    pywren.wait(futures)

    results = [f.result() for f in futures]
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]


    outdict = {'results' : results,
               'run_statuses' : run_statuses,
               'invoke_statuses' : invoke_statuses,
               'N' : N,
               'SLEEP_DUR' : SLEEP_DUR}
    pickle.dump(outdict, open("invocation.{}.pickle".format(N), 'wb'))
