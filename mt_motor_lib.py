# Definitions and variables for simulating a microtubule according to Peskin
# paper, and definitions and variables for simulating a motor (according to
# another Peskin paper!).

import numpy as np
import scipy

# Default simulation parameters
# -----------------------

# ``Chumakova constants''
# (Default parameters for Peskin MT growth simulation)
_chu_alpha = 1000
_chu_ab_ratio = 3.5
_chu_beta = _chu_ab_ratio * _chu_alpha
_chu_alpha_p = 4.0 # \alpha'
_chu_beta_p = 1.0 # \beta'
_chu_max_length = 600 # rounded cell diameter in dimers
_chu_wall = 'left' # or 'right'--orientation of tube


class Tube:

    # length - initial length; state - 0 for decaying, 
    # 1 for growing; alpha, beta, alpha_p, beta_p = \alpha, \beta, \alpha', 
    # \beta' in Peskin; max_length - max length of tube (in dimers).
    # wall is 'left' or 'right', denoting from which wall tube grows, and
    # diffusion_tube is where to send motors whose dimer is lost (set to None
    # if creating a diffusion tube)
    def __init__(self, length = 0, state = "decaying", 
            alpha = _chu_alpha, beta = _chu_beta, alpha_p = _chu_alpha_p,
        beta_p = _chu_beta_p, max_length = _chu_max_length,
            wall = _chu_wall, diffusion_tube = None, all_tubes = None):

        # Tube can't be length zero and be in growing regime...
        if (length == 0 and state == "growing"):
            print("Error in tube initilization: tube can't be length "
            "0 and state 1 (growing). Changing state to 0.")
            state = 0

        # Tube can't be longer than max_length
        if (length > max_length):
            printf("My TOOBE is too big // I am a banana")
            printf("Changing tube length to max_length")
            length = max_length

        self.length = length
        self.state = state

        self.alpha = alpha
        self.beta = beta
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.max_length = max_length 
        self.wall = wall
        self.diffusion_tube = diffusion_tube
        self.all_tubes = all_tubes # Used for recomputing rates for motor
                                    # whose dimer is lost.

        # A list of the motors on the tube, to know which motors to release
        # when a dimer is lost.
        self.motors = []

        # Check if tube is max length and therefore should be collapsing
        self.check_if_max_length()
        
        # The rate of any reaction in the current state--useful for
        # Gillespie algorithm.
        self.reaction_rate = 0.0
        self.update_reaction_rate()

    def __lt__(self, other):
        try:
            return self.length < other.length
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __gt__(self, other):
        try:
            return self.length > other.length
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented
            
    def __eq__(self, other):
        try:
            return self.length == other.length
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented
        

    # Reaction rate depends on the state and length of tube.
    # Later--try to redo everything using dictionaries?
    def update_reaction_rate(self):
        if (self.state == "decaying" and self.length == 0):
            self.reaction_rate = self.alpha_p
        elif (self.state == "decaying"):
            self.reaction_rate = self.alpha_p + self.beta
        elif (self.state == "growing" and self.length < self.max_length):
            self.reaction_rate = self.alpha + self.beta_p
        elif (self.state == "growing" and self.length == self.max_length):
            self.reaction_rate == self.alpha

    # Given an event happened, simulate which one it was.
    def simulate_event(self):
        # If tube is length 0, only possible event is tube growth.
        if (self.state == "decaying" and self.length == 0):
            self.state = "growing"
            self.length = self.length + 1
        # If tube length is >= max_length, do nothing
        #elif (self.length >= self.max_length):
        #    return
        # Tube is nonzero length, but in decaying regime.
        elif (self.state == "decaying"):
            alpha_p = self.alpha_p
            beta = self.beta
            state_change_prob = alpha_p / (alpha_p + beta)
            
            # Simulate which reaction happened.
            if (scipy.random.uniform(0,1) < state_change_prob):
                # State changed!
                self.state = "growing"
                self.length = self.length + 1
            else:
                # Tube shortened. :(
                self.length = self.length - 1
                # Throw away motors
                for motor in self.motors:
                    if (motor.position > self.length):
                        self.remove_motor(motor)
                        motor.state = '00'
                        motor.current_tube = self.diffusion_tube
                        if (self.wall == 'right'):
                            motor.position = self.max_length - \
                                    motor.position + 1
                        self.diffusion_tube.add_motor(motor)
                        motor.update_reaction_rate(self.all_tubes)                
        # Tube is in growing regime and not at the boundary
        elif (self.state == "growing" and self.length < self.max_length):
            alpha = self.alpha
            beta_p = self.beta_p
            state_change_prob = beta_p / (alpha + beta_p)

            # Simulate which reaction happened.
            if (scipy.random.uniform(0,1) < state_change_prob):
                # State changed!
                self.state = "decaying"
                self.length = self.length - 1
                # Throw away motors # check this
                for motor in self.motors:
                    if (motor.position > self.length):
                        self.remove_motor(motor)
                        motor.state = '00'
                        motor.current_tube = self.diffusion_tube
                        if (self.wall == 'right'):
                            motor.position = self.max_length - \
                                    motor.position + 1
                        self.diffusion_tube.add_motor(motor)
                        motor.update_reaction_rate(self.all_tubes)
            else:
                # Tube grew! *<(:)  <---party hat emoji
                self.length = self.length + 1
        # Tube in growing regime and at the boundary, so the only thing it
        # can do is transition to a depolymerizing state at the rate \alpha
        # (this was discussed with Lyuba on 26/02/18).
        elif (self.state == "growing" and self.length == self.max_length):
            # State changed!
            self.state = "decaying"
            self.length = self.length - 1
            # Throw away motors # check this
            for motor in self.motors:
                if (motor.position > self.length):
                    self.remove_motor(motor)
                    motor.state = '00'
                    motor.current_tube = self.diffusion_tube
                    if (self.wall == 'right'):
                        motor.position = self.max_length - \
                                motor.position + 1
                    self.diffusion_tube.add_motor(motor)
                    motor.update_reaction_rate(self.all_tubes)
        
        
        # Tube changed--need to update the reaction rate!
        #self.check_if_max_length()
        self.update_reaction_rate()


    # Check if a tube is max length, and if so, set all rates to 0
    # (tube is stable in this state)
    def check_if_max_length(self):
        # > case should never happen, unless max_length is changed 
        # manually--so don't do this!
        if (self.length >= self.max_length):
            self.alpha = 0.
            self.beta = 0.
            self.alpha_p = 0.
            self.beta_p = 0.

    def add_motor(self, new_motor):
        self.motors.append(new_motor)

    # Generously assume no duplicates...
    def remove_motor(self, motor):
        #motor.state = '00'
        #self.diffusion_tube.add_motor(motor)
        #motor.current_tube = self.diffusion_tube

        #motor.update_reaction_rate(self.all_tubes)
        self.motors.remove(motor)
    
    def end_position(self):
        if (self.wall == 'left'):
            return self.length
        if (tube.wall == 'right'):
            return self.max_length - self.length + 1
#
# Just one lonely motor...
#     ______
#    /|_||_\`.__
#   (   _    _ _\
#--~=`-(x)--(X)------
#
# simulated according to another paper by Peskin
#

class Motor:

    # Default simulation parameters
    # -----------------------
    # Made-up default constants

    _def_orientation = 1
    _def_alpha = 10000
    _def_beta_b = 12000 
    _def_beta_f = 4000
    _def_beta_bar = 1000 # not in Peskin, but in notes
    _def_front_P = 0.85
    _def_alpha_bar = 2000 # not in Peskin--reattachment rate for diffusing motor
    _def_r = 200 # Also not in Peskin--rate at which a ``diffusing'' motor
               # transitions from site to site (i.e. with rate r/2 to left)

    def __init__(self, current_tube, diffusion_tube, all_tubes, 
            alpha = _def_alpha, beta_b = _def_beta_b, beta_f = _def_beta_f,
            alpha_bar = _def_alpha_bar, beta_bar = _def_beta_bar, 
            front_P = _def_front_P, r = _def_r, position = 1, state = '00'):
        
        # Sanity check
        # To do: Put in checks that tube is long enough, etc.
        
        # alpha - rate of free head reattaching
        self.alpha = alpha
        # beta_b - rate of detachment for the back head
        self.beta_b = beta_b
        # beta_f - rate of detachment for the front head
        self.beta_f = beta_f
        # Reattachment rate for a motor that's drifting away
        self.alpha_bar = alpha_bar
        # beta_bar - rate of detachment for a head when second head is
        # already detached.
        self.beta_bar = beta_bar
        # front_P - probability that a head reattached in front (i.e. motor
        # made a foward step.
        self.front_P = front_P
        self.r = r
        # Coordinate of the forwardtmost attached leg, starting with 1
        self.position = position
        # State the tube is in. Possible states: 01, 11 (Peskin notation),
        # 00 (diffusing).
        self.state = state
        # Tube the motor is on. None if it's an idealized infinite tube.
        self.current_tube = current_tube
        # Use a full-grown tube (left-walled) as the diffusion matrix
        self.diffusion_tube = diffusion_tube
        # List of all the tubes (excluding diffusion tube)
        self.all_tubes = all_tubes
        # Add motor to tube
        self.current_tube.add_motor(self)

        # The rate of any reaction in the current state--useful for
        # Gillespie algorithm.
        self.reaction_rate = 0.0
        self.update_reaction_rate(self.all_tubes)

    # Leave functionality to send a subset of the tubes
    def update_reaction_rate(self, all_tubes):
        # Update reaction rate based on state
        
        # First do special cases.
        # E-cad is absorbed at the boundaries.
        
        # In principle, should check that the motor state isn't 11.
        if (self.position == 1):
            self.reaction_rate = 0.
        # In state 11, the motor reached final destiation; other states
        # have not reached the boundary yet.
        elif (self.position == 2 and self.state == '11'):
            self.reaction_rate = 0.
        elif (self.position == self.current_tube.max_length):
            self.reaction_rate = 0.
        # normal dynamics in the middle of the tube.
        else:
            if (self.state == '00'):
                # Assume reattachment rate is proportional to the number
                # of (non-diffusing) tubes
                r = self.r
                alpha_bar = self.alpha_bar
                n_local_tubes = n_tubes_at_location(self, all_tubes)
                
                self.reaction_rate = alpha_bar * n_local_tubes + r
            elif (self.state == '11'):
                self.reaction_rate = self.beta_f + self.beta_b
            elif (self.state == '01'):
                self.reaction_rate = self.alpha + self.beta_bar

    # Given an event happened, simulate what it was
    def simulate_event(self):
        if (self.reaction_rate == 0):
            return

        all_tubes = self.all_tubes
        position = self.position
        tube_length = self.current_tube.max_length
        alpha = self.alpha
        beta_b = self.beta_b
        beta_f = self.beta_f
        front_P = self.front_P
        alpha_bar = self.alpha_bar
        beta_bar = self.beta_bar
        r = self.r
        Z = self.reaction_rate
        
        if (position > 1 and position < tube_length):
            if (self.state == '00'):
                events = ['dec', 'inc', 'reattach']
                probabilities = [r / (2. * Z), r / (Z * 2.), 1. - r / Z]
                event = scipy.random.choice(events, p = probabilities)

                if (event == 'dec'):
                    self.position = position - 1
                elif (event == 'inc'):
                    self.position = position + 1
                elif (event == 'reattach'):
                    near_tubes = []
                    # Makes a list of all nearby tubes, and counts how many
                    n_near = find_near_tubes(self, all_tubes, near_tubes)
                    
                    # This is the WRONG BEHAVIOR!!! But it'll be rarely
                    # encountered, since we'll be working with bundles. Fix
                    # this in the final implementation.
                    # 03 March 2018 -- now fixed in the driver. Diffusing 
                    # motors' rates are updated every time an MT event occurs.
                    if (n_near != 0):     
                        # Detach and attach to a random one
                        random_tube = near_tubes[scipy.random.randint(n_near)]
                        self.current_tube.remove_motor(self)
                        self.current_tube = random_tube
                        self.current_tube.add_motor(self)
                        self.state = '01'
                        # Need to change coordinates w.r.t. flipped tube
                        # (diffusion matrix grows from left)
                        if (random_tube.wall == 'right'):
                            self.position = random_tube.max_length - position + 1

            elif (self.state == '01'):
                events = ['attach_b', 'attach_f', 'detach']
                probabilities = [alpha * (1. - front_P) / Z,
                        alpha * front_P / Z, 1. - alpha / Z]
                event = scipy.random.choice(events, p = probabilities)

                if (event == 'attach_b'):
                    self.state = '11'
                elif (event == 'attach_f'):
                    # Detach from MT if motor walks off tip
                    if (self.position == self.current_tube.length):
                        event = 'detach'
                    else:
                        self.state = '11'
                        self.position = position + 1
                if (event == 'detach'):
                    self.state = '00'
                    if (self.current_tube.wall == 'right'):
                        self.position = self.current_tube.max_length - position + 1
                    self.current_tube.remove_motor(self)
                    self.current_tube = self.diffusion_tube
                    self.current_tube.add_motor(self)

            elif (self.state == '11'):
                events = ['detach_b', 'detach_f']
                probabilities = [beta_b / Z, 1. - beta_b / Z]
                event = scipy.random.choice(events, p = probabilities)

                if (event == 'detach_b'):
                    self.state = '01'
                elif (event == 'detach_f'):
                    self.state = '01'
                    self.position = position - 1
        
                
        # If the motor is at the boundary and not in state 00, it's stuck!
        #elif (self.position == self.current_tube.max_length and 
        #        self.state != '00'):
        #    self.alpha = 0
        #    self.beta_f = 0
        #    self.beta_b = 0
        #    self.alpha_bar = 0
        #    self.beta_bar = 0

    
        self.update_reaction_rate(self.all_tubes)
        return event


    # Position with respect to the left wall
    def absolute_position(self):
        if (self.current_tube.wall == 'left'):
            return self.position
        elif (self.current_tube.wall == 'right'):
            return self.current_tube.max_length - self.position + 1


# Count show many tubes there near a motor 
def n_tubes_at_location(motor, all_tubes):
    position = None

    if (motor.current_tube.wall == 'left'):
        position = motor.position
    else:
        position = motor.current_tube.max_length - motor.position + 1

    n = 0

    for tube in all_tubes:
        if (tube.wall == 'left' and position <= tube.length):
            n = n + 1
        if (tube.wall == 'right' 
                and position > tube.max_length - tube.length):
            n = n + 1
    
    return n


def find_near_tubes(motor, all_tubes, near_tubes):
    position = None

    if (motor.current_tube.wall == 'left'):
        position = motor.position
    else:
        position = motor.current_tube.max_length - motor.position + 1

    n = 0
    near_tubes.clear()

    for tube in all_tubes:
        if (tube.wall == 'left' and position <= tube.length):
            n = n + 1
            near_tubes.append(tube)
        if (tube.wall == 'right' 
                and position > tube.max_length - tube.length):
            n = n + 1
            near_tubes.append(tube)
    
    return n

