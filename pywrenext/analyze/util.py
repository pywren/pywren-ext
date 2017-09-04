import pandas as pd


def job_df_from_futures(futures):
    """
    Extract all of the runtime info from a list of futures into
    a single pandas dataframe
    """

    invoke_statuses = [f.invoke_status for f in futures]
    run_statuses = [f.run_status for f in futures]

    invoke_df = pd.DataFrame(invoke_statuses)
    run_df = pd.DataFrame(run_statuses)
    del run_df['host_submit_time']
    results_df = pd.concat([run_df, invoke_df], axis=1)   
    return results_df

DEFAULT_FIELDS = ['host submit',
                  'job start',
                  'setup done',
                  'job done',
                  'results returned']

def normalize_times(results_df):
    """
    Normalize the times of a collection of job results, and return the 
    resulting dataframe
    """

    time_offset = np.min(results_df.host_submit_time)

    data = {'host submit' :  results_df.host_submit_time - time_offset, 
              'job start' : results_df.start_time - time_offset , 
              'setup done': results_df.start_time + results_df.setup_time - time_offset, 
              'job done' : results_df.end_time - time_offset, 
              'results returned': results_df.download_output_timestamp - time_offset
           }
    return pd.DataFrame(data)
