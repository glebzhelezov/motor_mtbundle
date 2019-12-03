import mt_motor_mc as mmm
import pickle
import time
import os

timestamp = str(time.time())
os.mkdir(timestamp)

# number of realizations in each experiment
n_realizations = 200
# each pickle contains this many experiments
batch_size = 40
# number of batches
n_batches = 2000000

labels = ['n_realizations', 'random_parameters', 'max_t', 'kinesin_times', 'dynein_times']

n_in_dir = 1000

for batch_id in range(n_batches):
    # so as to not clog everything up
    if (batch_id % n_in_dir == 0):
        directory_id = int(batch_id/n_in_dir)
        os.mkdir("{}/{}".format(timestamp, directory_id))
        path = "{}/{}".format(timestamp, directory_id)
    
    experiments = []
    for param_set_id in range(batch_size):
        random_parameters = mmm.generate_parameters()
        max_t = 200*mmm.simulation_timescale(random_parameters)

        result = mmm.generate_many_samples(
            n_realizations,
            *random_parameters,
            -1,
            max_t,
        )
        
        experiments.append([
            n_realizations,
            random_parameters,
            max_t,
            result[0],
            result[1],
        ])
    with open("{}/{}.p".format(path, batch_id), 'wb') as file:
        pickle.dump([labels, experiments], file)
