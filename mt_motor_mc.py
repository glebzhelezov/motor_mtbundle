import mt_motor_lib st mml 
import numpy as np
import pickle
from joblib import Parallel, delayed
import multiprocessing
import time
import os
import mt_equilibirum as mte

def simulate_realization_pair(
    K,
    N_L,
    N_R,
    alpha,
    beta,
    alpha_prime,
    beta_prime,
    P_front,
    beta_b,
    beta_f,
    omega,
    eta,
    gamma,
    beta_bar,
    initial_position = -1,
    t_max = 1000,
    return_trajectories = False,
    burn_in = False,
):
    # set initial
    if (initial_position == -1):
        initial_position = K/2
    if (initial_position < -1 or initial_position >= K):
        print("Motor not in cell")
        return
    
    initial_position = int(initial_position)
    
    # Gillepsie on these
    microtubules = []
    motors = []
    
    # Motor trajectory
    motor_position = []
    motor_abs_time = -1
    # Mirror motor trajectory
    mirror_motor_position = []
    mirror_motor_abs_time = -1
    # Trajcetories of MTs; lists of lists.
    left_mt_lengths = []
    right_mt_lengths = []

    # Common time vector (we update all lists at every Gillepsie jump)
    times = [0.]
    
    # Set up MT/motor system
    # Motors diffuse on this
    diffusion_tube = mml.Tube(length = K, max_length = K)
    
    # Sample from equilibrium distribution for MT lengths
    p_vals, q_vals = mte.compute_probabilities(
        K, alpha, beta_prime, beta, alpha_prime
    )

    mt_choices = list(
        zip(
            ['growing' for _ in range(K+1)] + ['decaying' for _ in range(K+1)],
            list(range(K+1)) + list(range(K+1))
        )
    )

    mt_probs = list(p_vals) + list(q_vals)
    
    # Create MTs growing from the left
    for _ in range(N_L):
        initial_index = np.random.choice(range(len(mt_choices)), p=mt_probs)
        initial_state, initial_length = mt_choices[initial_index]
        microtubules.append(
            mml.Tube(
                length = initial_length,
                state = initial_state,
                alpha = alpha,
                beta = beta,
                alpha_p = alpha_prime,
                beta_p = beta_prime,
                max_length = K,
                wall = 'left',
                diffusion_tube = diffusion_tube,
                all_tubes = microtubules,
            )
        )
        
        left_mt_lengths.append([])
        left_mt_lengths[-1].append(initial_length)
        
    # Create MTs growing from the right
    for _ in range(N_R):
        initial_index = np.random.choice(range(len(mt_choices)), p=mt_probs)
        initial_state, initial_length = mt_choices[initial_index]
        microtubules.append(
            mml.Tube(
                length = initial_length,
                state = initial_state,
                alpha = alpha,
                beta = beta,
                alpha_p = alpha_prime,
                beta_p = beta_prime,
                max_length = K,
                wall = 'right',
                diffusion_tube = diffusion_tube,
                all_tubes = microtubules,
            )
        )
        
        right_mt_lengths.append([])
        right_mt_lengths[-1].append(initial_length)

    # Simulate the MTs so they're approximately in steady state
    if burn_in is not False:
        t = 0
        t_burnin = 5*min([alpha, beta, beta_prime, alpha_prime])**(-1)
        while (t < t_burnin):
            # Reaction rate for the MTs
            microtubules_rate = 0.

            for tube in microtubules:
                microtubules_rate += tube.reaction_rate

            rate = microtubules_rate

            # Add waiting time.
            t += np.random.exponential(1. / rate)

            # This forms a list of Motor and Tube objects
            possible_events = microtubules

            probabilities = []

            # Forms a list of the probability that a given object reacted
            for event in possible_events:
                probabilities.append(event.reaction_rate / rate)

            # Choose the object with an appropriate probability
            event = np.random.choice(possible_events, p = probabilities)

            # Simulate the event
            done_event = event.simulate_event()

        
    # Create a motor
    original_motor = mml.Motor(
            diffusion_tube,
            diffusion_tube,
            microtubules,
            alpha = omega, # reattachment of free leg
            beta_b = beta_b, # back leg detachment
            beta_f = beta_f, # front leg detachment
            alpha_bar = gamma, # reattachment rate
            beta_bar = beta_bar, # second leg detachment rate
            front_P = P_front, # prob. of attaching in front
            r = eta, # diffusion rate
            position = initial_position, # start in the middle
            state = '00', # start in diffusing state
    )
      
    # This is the motor walking the other way
    mirror_motor = mml.Motor(
            diffusion_tube,
            diffusion_tube,
            microtubules,
            alpha = omega, # reattachment of free leg
            beta_b = beta_f, # back leg detachment
            beta_f = beta_b, # front leg detachment
            alpha_bar = gamma, # reattachment rate
            beta_bar = beta_bar, # second leg detachment rate
            front_P = 1-P_front, # prob. of attaching in front
            r = eta, # diffusion rate
            position = initial_position, # start in the middle
            state = '00', # start in diffusing state
    )
    
    # Save the motors
    motors.append(mirror_motor)
    motors.append(original_motor)
    # Save their initial positions
    motor_position.append(original_motor.absolute_position())
    mirror_motor_position.append(mirror_motor.absolute_position())
    
    # Simulate both motors until both are absorbed.
    t = 0
    while (t < t_max):
        # Reaction rate for the MTs
        microtubules_rate = 0.
        
        for tube in microtubules:
            microtubules_rate += tube.reaction_rate
        
        # Reaction rate for the motors
        motors_rate = 0.
        
        for motor in motors:
            motors_rate += motor.reaction_rate
        
        # Break if no motors are moving!
        if (motors_rate == 0):
            break
        
        rate = microtubules_rate + motors_rate
    
        # Add waiting time.
        t += np.random.exponential(1. / rate)
        
        # This forms a list of Motor and Tube objects
        possible_events = motors + microtubules

        probabilities = []

        # Forms a list of the probability that a given object reacted
        for event in possible_events:
            probabilities.append(event.reaction_rate / rate)

        # Choose the object with an appropriate probability
        event = np.random.choice(possible_events, p = probabilities)

        # Simulate the event
        done_event = event.simulate_event()
        
        # Kind of ugly, but it works...
        if (isinstance(event, mml.Motor)):
            if event.reaction_rate == 0:
                if event == original_motor:
                    motor_abs_time = t
                if event == mirror_motor:
                    mirror_motor_abs_time = t
                motors.remove(event)
            # we're done with this motor!
        
        # If last event was a MT (de)polymerizing, the rates at which the
        # diffusing motors are reattaching may have changed, since reattachment
        # rate is proportional to hte number of nearby MTs. Later, in the C
        # implementation, this should all be inside of an MT bundle class which
        # keeps track of which MT is available where, etc.--this will reduce the
        # computation considerably.
        if (isinstance(event, mml.Tube)):
            for motor in diffusion_tube.motors:
                motor.update_reaction_rate(microtubules)
        
        # For generating trajectories, etc.
        #for i in range(n_motors):
        #    motor_positions[i].append(motors[i].absolute_position())
        #    motor_states[i].append(motors[i].state)
        motor_position.append(original_motor.absolute_position())
        mirror_motor_position.append(mirror_motor.absolute_position())
        
        for i in range(N_L):
            left_mt_lengths[i].append(microtubules[i].length)
        for i in range(N_R):
            right_mt_lengths[i].append(microtubules[N_L + i].length)
        
        times.append(t)
        
        
    if (return_trajectories == True):
        return times, motor_position, mirror_motor_position, left_mt_lengths, right_mt_lengths

    return motor_abs_time, mirror_motor_abs_time

def simulate_realization_single(
    K,
    N_L,
    N_R,
    alpha,
    beta,
    alpha_prime,
    beta_prime,
    P_front,
    beta_b,
    beta_f,
    omega,
    eta,
    gamma,
    beta_bar,
    initial_position = -1,
    t_max = 1000,
    return_trajectories = False,
    burn_in = False,
):
    # set initial
    if (initial_position == -1):
        initial_position = K/2
    if (initial_position < -1 or initial_position >= K):
        print("Motor not in cell")
        return
    
    initial_position = int(initial_position)
    
    # Gillepsie on these
    microtubules = []
    motors = []
    
    # Motor trajectory
    motor_position = []
    motor_abs_time = -1

    # Trajcetories of MTs; lists of lists.
    left_mt_lengths = []
    right_mt_lengths = []

    # Common time vector (we update all lists at every Gillepsie jump)
    times = [0.]
    
    # Set up MT/motor system
    # Motors diffuse on this
    diffusion_tube = mml.Tube(length = K, max_length = K)

    # Sample from equilibrium distribution for MT lengths
    p_vals, q_vals = mte.compute_probabilities(
        K, alpha, beta_prime, beta, alpha_prime
    )
    
    # Sample initial MT length from the equilibrium distribution
    mt_choices = list(
        zip(
            ['growing' for _ in range(K+1)] + ['decaying' for _ in range(K+1)],
            list(range(K+1)) + list(range(K+1))
        )
    )

    mt_probs = list(p_vals) + list(q_vals)
    # Create MTs growing from the left
    for _ in range(N_L):
        initial_index = np.random.choice(range(len(mt_choices)), p=mt_probs)
        initial_state, initial_length = mt_choices[initial_index]
        
        microtubules.append(
            mml.Tube(
                length = initial_length,
                state = initial_state,
                alpha = alpha,
                beta = beta,
                alpha_p = alpha_prime,
                beta_p = beta_prime,
                max_length = K,
                wall = 'left',
                diffusion_tube = diffusion_tube,
                all_tubes = microtubules,
            )
        )
        
        left_mt_lengths.append([])
        left_mt_lengths[-1].append(initial_length)
        
    # Create MTs growing from the right
    for _ in range(N_R):
        initial_index = np.random.choice(range(len(mt_choices)), p=mt_probs)
        initial_state, initial_length = mt_choices[initial_index]
        
        microtubules.append(
            mml.Tube(
                length = initial_length,
                state = initial_state,
                alpha = alpha,
                beta = beta,
                alpha_p = alpha_prime,
                beta_p = beta_prime,
                max_length = K,
                wall = 'right',
                diffusion_tube = diffusion_tube,
                all_tubes = microtubules,
            )
        )
        
        right_mt_lengths.append([])
        right_mt_lengths[-1].append(initial_length)
 
    # Simulate the MTs so they're approximately in steady state
    if burn_in is not False:
        t = 0
        t_burnin= 2000*max([alpha, beta, beta_prime, alpha_prime])**(-1)
        while (t < t_burnin):
            # Reaction rate for the MTs
            microtubules_rate = 0.

            for tube in microtubules:
                microtubules_rate += tube.reaction_rate

            rate = microtubules_rate

            # Add waiting time.
            t += np.random.exponential(1. / rate)

            # This forms a list of Motor and Tube objects
            possible_events = microtubules

            probabilities = []

            # Forms a list of the probability that a given object reacted
            for event in possible_events:
                probabilities.append(event.reaction_rate / rate)

            # Choose the object with an appropriate probability
            event = np.random.choice(possible_events, p = probabilities)

            # Simulate the event
            done_event = event.simulate_event()

    
    # Create a motor
    original_motor = mml.Motor(
            diffusion_tube,
            diffusion_tube,
            microtubules,
            alpha = omega, # reattachment of free leg
            beta_b = beta_b, # back leg detachment
            beta_f = beta_f, # front leg detachment
            alpha_bar = gamma, # reattachment rate
            beta_bar = beta_bar, # second leg detachment rate
            front_P = P_front, # prob. of attaching in front
            r = eta, # diffusion rate
            position = initial_position, # start in the middle
            state = '00', # start in diffusing state
    )
      
    
    # Save the motors
    motors.append(original_motor)
    # Save their initial positions
    motor_position.append(original_motor.absolute_position())
    
    # Simulate both motors until both are absorbed.
    t = 0
    while (t < t_max):
        # Reaction rate for the MTs
        microtubules_rate = 0.
        
        for tube in microtubules:
            microtubules_rate += tube.reaction_rate
        
        # Reaction rate for the motors
        motors_rate = 0.
        
        for motor in motors:
            motors_rate += motor.reaction_rate
        
        # Break if no motors are moving!
        if (motors_rate == 0):
            break
        
        rate = microtubules_rate + motors_rate
    
        # Add waiting time.
        t += np.random.exponential(1. / rate)
        
        # This forms a list of Motor and Tube objects
        possible_events = motors + microtubules

        probabilities = []

        # Forms a list of the probability that a given object reacted
        for event in possible_events:
            probabilities.append(event.reaction_rate / rate)

        # Choose the object with an appropriate probability
        event = np.random.choice(possible_events, p = probabilities)

        # Simulate the event
        done_event = event.simulate_event()
        
        # Kind of ugly, but it works...
        if (isinstance(event, mml.Motor)):
            if event.reaction_rate == 0:
                if event == original_motor:
                    motor_abs_time = t
                motors.remove(event)
            # we're done with this motor!
        
        # If last event was a MT (de)polymerizing, the rates at which the
        # diffusing motors are reattaching may have changed, since reattachment
        # rate is proportional to hte number of nearby MTs. Later, in the C
        # implementation, this should all be inside of an MT bundle class which
        # keeps track of which MT is available where, etc.--this will reduce the
        # computation considerably.
        if (isinstance(event, mml.Tube)):
            for motor in diffusion_tube.motors:
                motor.update_reaction_rate(microtubules)
        
        # For generating trajectories, etc.
        #for i in range(n_motors):
        #    motor_positions[i].append(motors[i].absolute_position())
        #    motor_states[i].append(motors[i].state)
        motor_position.append(original_motor.absolute_position())
        
        for i in range(N_L):
            left_mt_lengths[i].append(microtubules[i].length)
        for i in range(N_R):
            right_mt_lengths[i].append(microtubules[N_L + i].length)
        
        times.append(t)
        
        
    if (return_trajectories == True):
        return times, motor_position, left_mt_lengths, right_mt_lengths

    return motor_abs_time
    
def generate_many_samples_pair(
    n_experiments,
    K,
    N_L,
    N_R,
    alpha,
    beta,
    alpha_prime,
    beta_prime,
    P_front,
    beta_b,
    beta_f,
    omega,
    eta,
    gamma,
    beta_bar,
    initial_position,
    t_max,
    burn_in = False,
):
    args_to_pass = [        
        K,
        N_L,
        N_R,
        alpha,
        beta,
        alpha_prime,
        beta_prime,
        P_front,
        beta_b,
        beta_f,
        omega,
        eta,
        gamma,
        beta_bar,
        initial_position,
        t_max,
        burn_in,
    ]
    
    original_arrivals = []
    mirror_arrivals = []
    
    with Parallel(n_jobs=-1, backend='loky') as parallel:
        results = parallel(delayed(simulate_realization_pair)(*args_to_pass)
                           for _ in range(n_experiments)
                    )
    
    original_arrivals = [res[0] for res in results]
    mirror_arrivals = [res[1] for res in results]
    
    #for _ in range(n_experiments):
    #    a, b = simulate_realization(*args_to_pass)
    #    original_arrivals.append(a)
    #    mirror_arrivals.append(b)
    
    return original_arrivals, mirror_arrivals


def generate_many_samples_single(
    n_experiments,
    K,
    N_L,
    N_R,
    alpha,
    beta,
    alpha_prime,
    beta_prime,
    P_front,
    beta_b,
    beta_f,
    omega,
    eta,
    gamma,
    beta_bar,
    initial_position,
    t_max,
    burn_in = False,
):
    args_to_pass = [        
        K,
        N_L,
        N_R,
        alpha,
        beta,
        alpha_prime,
        beta_prime,
        P_front,
        beta_b,
        beta_f,
        omega,
        eta,
        gamma,
        beta_bar,
        initial_position,
        t_max,
        False, # don't return trajectories
        burn_in
    ]
    
    original_arrivals = []
    
    with Parallel(n_jobs=-1, backend='loky') as parallel:
        results = parallel(delayed(simulate_realization_single)(*args_to_pass)
                           for _ in range(n_experiments)
                    )
    
    original_arrivals = results
    
    #for _ in range(n_experiments):
    #    a, b = simulate_realization(*args_to_pass)
    #    original_arrivals.append(a)
    #    mirror_arrivals.append(b)
    
    return original_arrivals



def mean_velocity(beta_b, beta_f, omega, p):
    """Average run velocity for a motor."""
    beta = beta_f + beta_b
    delta = beta_b - beta_f
    
    velocity = omega/(2*(omega+beta)) * (delta + 2*(p-0.5)*beta)
    
    return velocity


# random parameters for use
def generate_parameters(
    K_min = 50,
    K_max = 200,
    N_L_min = 3,
    N_L_max = 5,
    N_R_min = 3,
    N_R_max = 5,
    alpha = 1000, #this just provides a scale
    beta_min = 2e3,
    beta_max = 6e3,
    alpha_prime_min = 2,
    alpha_prime_max = 10,
    p_min = 0,
    p_max = 1,
    beta_b_min = 1000,
    beta_b_max = 5e4,
    beta_f_min = 1000,
    beta_f_max = 5e4,
    omega_min = 1000,
    omega_max = 5e4,
    eta_min = 150,
    eta_max = 800,
):
    """Generate parameters satisfying the paper. Parameters for which max/min unavailable
    depend on the other parameter selections."""
    
    # These are sampled according to the bounds above
    K = np.random.randint(K_min, K_max+1)
    N_L = np.random.randint(N_L_min, N_L_max+1)
    N_R = np.random.randint(N_R_min, N_R_max+1)
    #alpha = alpha
    beta = np.random.uniform(beta_min, beta_max)
    alpha_prime = np.random.uniform(alpha_prime_min, alpha_prime_max)
    beta_prime = np.random.uniform(alpha_prime/10, alpha_prime/2)
    eta = np.random.uniform(eta_min, eta_max)
    gamma = np.random.uniform(eta/10, eta/2)
        
    #these need to satisfy a relation
    while True:
        beta_b = np.random.uniform(beta_b_min, beta_b_max)
        beta_f = np.random.uniform(beta_f_min, beta_f_max)
        omega = np.random.uniform(omega_min, omega_max)
        p = np.random.uniform(p_min, p_max)
        
        v = mean_velocity(beta_b, beta_f, omega, p)
        mirror_v = mean_velocity(beta_f, beta_b, omega, 1-p)
        
        # accept parameter choices iff they force kinesin and dynein run
        # at approriate speeds.
        # here for some values of alpha you might run into trouble
        #if (alpha < v and v < 5*alpha):
        #    if (beta < -mirror_v and -mirror_v < 5*beta):
        #        break
        if (alpha < v):
            if (beta < -mirror_v):
                break
    
    # fall-off rate is less than stepping rate
    beta_bar = np.random.uniform(0, omega/5)
    
    random_parameters = [
        K,
        N_L,
        N_R,
        alpha,
        beta,
        alpha_prime,
        beta_prime,
        p,
        beta_b,
        beta_f,
        omega,
        eta,
        gamma,
        beta_bar,
    ]
    
    return random_parameters


def simulation_timescale(random_parameters):
    """MC sim. should be 100-500x this to definitely see a big diff. between kinesin and dynein"""
    beta_b = random_parameters[8]
    beta_f = random_parameters[9]
    omega = random_parameters[10]
    p = random_parameters[11]
    K = random_parameters[0]
    alpha = random_parameters[3]
    speed = mean_velocity(beta_b, beta_f, omega, p)
    
    timescale = 100*(0.5*K/alpha + K/(2*speed))
    
    return timescale


