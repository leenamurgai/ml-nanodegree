import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from math import exp

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    res_state_action = open('./state_action.csv', 'w+')
    res_on_time      = open('./on_time.csv', 'w+')
    res_q_values     = open('./q_values.txt', 'w+')
    res_perf_stats   = open('./perf_stats.csv', 'w+')

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.AgentState = namedtuple('AgentState', ['desired_action', 'light', 'oncoming_is_fwd_or_rt', 'left_is_fwd', 'late'])
        self.best_action = random.choice(self.env.valid_actions) # random action to start
        self.time = 0
        self.gamma = 0.5 # discount factor for neighbouring q-value
        self.initial_q = 2.1 # q table values are all set to this value to start
        self.q_table = {} # q table values are stored in a dictionary
        # self.best_action = random.choice(self.env.valid_actions) # random action
        self.n_trip = 0 # number of trips
        self.n_arr_in_time = 0 # number of trips where the cab arrived in time
        self.n_moves = 0
        self.initial_deadline = 0
        self.valid_actions = [None, 'forward', 'left', 'right']
        print >> self.res_state_action, "trip,time,deadline,light,oncoming_is_fwd_or_rt,left_is_fwd,next waypoint,action,reward,net reward"
        print >> self.res_on_time, "number trip,number arrived on time,deadline,number of moves"
        print >> self.res_perf_stats, "num trip,num in time,num moves,num penalites,num wrong way legal,num None,num next waypoint legal,percentage deadline used,avg reward"

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.n_trip += 1
        self.net_reward = 0
        self.n_moves = 0
        self.n_penalties = 0
        self.n_wrong_way_legal = 0
        self.n_none = 0
        self.n_nxtwpt_legal = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.AgentState( desired_action = self.next_waypoint,
                                      light = inputs['light'],
                                      oncoming_is_fwd_or_rt = inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right',
                                      left_is_fwd = inputs['left'] == 'forward',
                                      late = deadline < 0 )

        # TODO: Select action according to your policy

        # 1. Implement a basic driving agent
        # self.best_action = random.choice(self.env.valid_actions) # random action

        # 3. Implement Q-learning
        action = self.best_action # Take the best_action calculated on the previous iteration, this is random to start

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.net_reward = self.net_reward + reward

        # Note the deadline at the begining of every trial
        if self.n_moves == 0:
            self.initial_deadline = deadline
        # Update n_moves
        self.n_moves += 1
        if reward == -1 or reward == 9:
            self.n_penalties += 1
        if reward == 0.5 or reward == 10.5:
            self.n_wrong_way_legal += 1
        if reward == 1:
            self.n_none += 1
        if reward == 2 or reward == 12:
            self.n_nxtwpt_legal += 1

        # print some results relating to the state, actions and rewards
        #print "trip = {} time = {}, deadline = {}, light = {}, oncoming_is_fwd_or_rt = {}, left_is_fwd = {} next_waypoint = {}, action = {},reward = {}, net_reward = {}".format(
        #self.n_trip, self.time, deadline,inputs['light'], self.state.oncoming_is_fwd_or_rt, self.state.left_is_fwd, self.next_waypoint, action, reward, self.net_reward)
        print >> self.res_state_action, "{},{},{},{},{},{},{},{},{},{}".format(self.n_trip, self.time, deadline,
        inputs['light'], self.state.oncoming_is_fwd_or_rt, self.state.left_is_fwd, self.next_waypoint, action, reward, self.net_reward)

        # Update number of trips where we arrived on time
        if reward >= 9:
            self.n_arr_in_time += 1

        # At the end of a trial print some stats about how well our agent is doing at getting to the destination
        if reward >= 9 or deadline == 0:
            print "LearningAgent.update(): number trips = {}, number arrived on time = {}, initial deadline = {}, number of moves = {}".format(
                                                            self.n_trip, self.n_arr_in_time, self.initial_deadline, self.n_moves)
            print >> self.res_on_time, "{},{},{},{}".format(self.n_trip, self.n_arr_in_time, self.initial_deadline, self.n_moves)
            pct_deadline = float(self.n_moves)/self.initial_deadline
            avg_reward = 0
            if reward >= 9:
                avg_reward = float(self.net_reward - 10)/self.n_moves
            else:
                avg_reward = float(self.net_reward)/self.n_moves
            print >> self.res_perf_stats, "{},{},{},{},{},{},{},{},{}".format(self.n_trip, self.n_arr_in_time, self.n_moves,
            self.n_penalties, self.n_wrong_way_legal, self.n_none, self.n_nxtwpt_legal, pct_deadline, avg_reward)

        # Update time having taken an action
        self.time += 1

        # TODO: Learn policy based on state, action, reward

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Sense the new state we are in having taken an action and reward
        new_state = self.AgentState( desired_action = self.next_waypoint,
                                     light = inputs['light'],
                                     oncoming_is_fwd_or_rt = inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right',
                                     left_is_fwd = inputs['left'] == 'forward',
                                     late = deadline < 0 )

        # 3. Implement Q-learning
        # start by picking best_action randomly
        self.best_action = random.choice(self.env.valid_actions)
        best_q = self.get_q_table_value(new_state, self.best_action)
        # For our new state, find the action (best_action) with the largest q-value (best_q) - Greedy learner
        # randomise the order of valid_actions to avoid disproportionately picking actions earlier in the list
        random.shuffle(self.valid_actions)
        for a in self.valid_actions:
            q_value = self.get_q_table_value(new_state, a)
            #print "LearningAgent.update(): action = {}, q value = {}".format(a, q_value)
            print >> self.res_q_values, "action = {}, q value = {}".format(a, q_value)
            # Find best_action and best_q
            if best_q < q_value:
                self.best_action = a
                best_q = q_value

        # Calculate the new q-value for our previous state and action using best_q
        q_value = ( 1.0-self.alpha(self.time) )*self.get_q_table_value(self.state, action) + self.alpha(self.time)*( reward + self.gamma*best_q )
        self.set_q_table_value(self.state, action, q_value)

        # Print some info about the chosen action and new q-value
        #print "LearningAgent.update(): num trips = {}, num arrived on time = {}, time = {}, desired action = {}, best action = {}, new q-value = {}\n".format(
        #                         self.n_trip, self.n_arr_in_time, self.time, self.next_waypoint, self.best_action, q_value)
        print >> self.res_q_values, "num trips = {}, num arrived on time = {}, time = {}, desired action = {}, best action = {}, new q-value = {}\n".format(
                                 self.n_trip, self.n_arr_in_time, self.time, self.next_waypoint, self.best_action, q_value)

    # Learning rate
    def alpha(self, time):
        return 1.0/time

    # Populate the q table as the (state, action) space is explored rather than initialising for all possible (state, action) pairs
    def get_q_table_value(self, state, action):
        if self.q_table.get((state, action)) == None:
            self.q_table[(state, action)] = self.initial_q
        return self.q_table[(state, action)]

    #
    def set_q_table_value(self, state, action, value):
        self.q_table[(state, action)] = value

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
