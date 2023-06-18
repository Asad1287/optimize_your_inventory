import numpy as np
import simpy


from discrete_event_simulation.single.simulation_classes import *
from typing import List, Tuple, Callable
import random
from discrete_event_simulation.single.des_single_continous import *

import math

def cost_function(ROP:float, EOQ:float,Setting:object) -> float:

    simulation_time = Setting.TIME
    purchase_cost = Setting.purchase_cost
    selling_price = Setting.selling_price
    
    holding_cost_unit = Setting.holding_cost_unit 
    ordering_cost_unit = Setting.ordering_cost
    other_costs = Setting.other_costs
    lead_time_mean = Setting.lead_time_mean
    lead_time_std = Setting.lead_time_std
    delivery_batches = Setting.delivery_batches
    daily_demand_mean =Setting.daily_demand_mean
    daily_demand_std =Setting.daily_demand_std
    review_period = Setting.review_period
    backlog_cost_unit = Setting.backlog_cost_unit
    safety_stock = Setting.safety_stock
    balance = Setting.balance
    constants = Constants(ROP, EOQ, purchase_cost, selling_price, holding_cost_unit, ordering_cost_unit, other_costs, lead_time_mean, lead_time_std, delivery_batches, daily_demand_mean, daily_demand_std, review_period, backlog_cost_unit, safety_stock, balance)
    variables = Variables(constants)
    functions = ExternalFunctions()
    env = simpy.Environment()
    simulation = Inventory_Simulation(env, constants, variables, functions)
    env.process(simulation.runner_setup())
    env.process(simulation.observe())
    env.run(until=simulation_time)
    return simulation.calculate_total_cost()

def simulated_annealing_optimizer(n_iterations:int, Setting:object, initial_temp:float=1000, cool_down_rate:float=0.9) -> Tuple[float, float, float]:
    
    # Initialize random solution (ROP and EOQ) within a given range
    current_ROP = random.randint(4, 100)
    current_EOQ = random.randint(2, 100)
    current_cost = cost_function(current_ROP, current_EOQ, Setting)
    
    best_ROP = current_ROP
    best_EOQ = current_EOQ
    best_cost = current_cost

    # Start with a high initial temperature (initial_temp) and decrease it over time (cool_down_rate)
    temp = initial_temp

    for _ in range(n_iterations):
        # Create a new candidate solution
        new_ROP = random.randint(4, 100)
        new_EOQ = random.randint(2, 100)
        new_cost = cost_function(new_ROP, new_EOQ, Setting)

        # Check if new solution is better, if not accept it probabilistically based on temperature
        if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / temp):
            current_ROP, current_EOQ, current_cost = new_ROP, new_EOQ, new_cost
            
            # Keep track of the best solution found
            if current_cost < best_cost:
                best_ROP, best_EOQ, best_cost = current_ROP, current_EOQ, current_cost

        # Reduce the temperature (Cooling down)
        temp *= cool_down_rate

    return best_ROP, best_EOQ, best_cost

# Run the Simulated Annealing optimizer
best_ROP, best_EOQ, best_cost = simulated_annealing_optimizer(1000, Setting)
print(f'Best ROP: {best_ROP}')
print(f'Best EOQ: {best_EOQ}')
print(f'Best cost: {best_cost}')