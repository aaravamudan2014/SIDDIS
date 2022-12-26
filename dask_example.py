from dask_jobqueue import PBSCluster
import dask
from math import sqrt
from dask.distributed import Client
import time
import random
import timeit

cluster = PBSCluster()
cluster.scale(jobs=10)    # Deploy ten single-node jobs
client = Client(cluster)  # Connect this local process to remote workers

def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)

num_simulations = 200
start = timeit.default_timer()

# serial for loop
for i in range(num_simulations):
    costly_simulation([1,2,3,4])

stop = timeit.default_timer()
print('Time before dask: ', stop - start)  


# parallelized for loop
start = timeit.default_timer()
lazy_results = []
for i in range(num_simulations):
    lazy_result = dask.delayed(costly_simulation)([1,2,3,4])
    lazy_results.append(lazy_result)

dask.compute(*lazy_results)
stop = timeit.default_timer()
print('Time after dask: ', stop - start)  
