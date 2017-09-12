import pywren
import time
import pickle
import boto3
import time
from pywrenext.progwait import progwait
def sleep_test(x):
    time.sleep(x)
    return x + 1

if __name__ == "__main__":
    
    t1 = time.time()

    N = 1500
    SLEEP_DUR = 50


    wrenexec = pywren.default_executor()
    futures = wrenexec.map(sleep_test, [SLEEP_DUR] * N)

    progwait(futures)

    results = [f.result() for f in futures]
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    t2 = time.time()

    outdict = {'results' : results,
               'futures' : futures, 
               'run_statuses' : run_statuses,
               'invoke_statuses' : invoke_statuses,
               'N' : N,
               'time' : t2-t1,
               'SLEEP_DUR' : SLEEP_DUR}
    pickle.dump(outdict, open("invocation.{}.pickle".format(N), 'wb'))
