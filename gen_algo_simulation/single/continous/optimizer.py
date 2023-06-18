import sys
sys.path.append('/mnt/d/Portfolio/Inventory_Optimization/inventory_optimizer_app')
import numpy as np
import pandas as pd
from gen_algo_simulation.single.Demand import *
from gen_algo_simulation.single.Leadtime  import *
from gen_algo_simulation.single.continous.simulation import *


from numba import njit
from numba import njit
import numpy as np
from gen_algo_simulation.simulation_settings import *
from numba import njit
import numpy as np


GENERATIONS = None
POPULATION_SIZE = None
MUTATION_RATE = None
TIME = None
PARAMETER_ONE_LIMITS = None
PARAMETER_TWO_LIMITS = None

GENERATIONS = Setting.GENERATIONS.value
POPULATION_SIZE = Setting.POPULATION_SIZE.value
MUTATION_RATE = Setting.MUTATION_RATE.value
TIME = Setting.TIME.value
PARAMETER_ONE_LIMITS = Setting.PARAMETER_ONE_LIMITS.value
PARAMETER_TWO_LIMITS = Setting.PARAMETER_TWO_LIMITS.value

def setup_optimizer(ROP:int,EOQ:int,time:int,demand:List[float],lead_times:List[float],cost_per_transcation:float,hold_cost_per_unit:float,backlog_cost:float,sale_price:float,cost_price:float,tax_rate:float):
    return ROP,EOQ,time,demand,lead_times,cost_per_transcation,hold_cost_per_unit,backlog_cost,sale_price,cost_price,tax_rate



@njit
def cumsum(list_x:List[float]) -> List[float]:
    for i in range(1,len(list_x)):
        list_x[i] = list_x[i-1] + list_x[i]
    return list_x

@njit
def random_choice_numba(a, p):
    return a[np.searchsorted(cumsum(p), np.random.random(), side="right")]

@njit
def init_population(pop_size):
    population = np.zeros((pop_size, 2), dtype=np.int64)
    for i in range(pop_size):
        population[i, 0] = np.random.randint(1, 13)  # Review period
        population[i, 1] = np.random.randint(1, 101)  # Safety stock
    return population

# Calculate fitness (the lower the cost, the higher the fitness)
@njit
def fitness(individual, time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate):
    cost, _, _, _ = simulation_process_single_item(individual[0], individual[1], time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate) 
    if cost == 0:
        return np.inf  # return a large fitness value for zero cost
    else:
        return 1 / (1+cost)
# Perform selection
@njit
def selection(population, fitnesses):
    parents = []
    for _ in range(POPULATION_SIZE // 2):
        # Select two parents randomly, with probability proportional to their fitness
        parent1 = np.random.choice(population, p=fitnesses)
        parent2 = np.random.choice(population, p=fitnesses)
        parents.append(parent1)
        parents.append(parent2)
    return parents

# Perform crossover
@njit
def crossover(parent1, parent2):
    crossover_index = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossover_index], parent2[crossover_index:]))
    child2 = np.concatenate((parent2[:crossover_index], parent1[crossover_index:]))
    return child1, child2

# Perform mutation
@njit
def mutation(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        index = np.random.randint(0, 2)
        if index == 0: # Review period
            individual[index] = np.random.randint(PARAMETER_ONE_LIMITS[0], PARAMETER_ONE_LIMITS[1] + 1)
        else: # Safety stock
            individual[index] = np.random.randint(PARAMETER_TWO_LIMITS[0], PARAMETER_TWO_LIMITS[1] + 1)
    return individual

# Genetic algorithm
@njit
def find_max_list(list):
    max = list[0]
    for i in list:
        if i > max:
            max = i
    return max
@njit
def find_argmax_list(list):
    max = list[0]
    argmax = 0
    for i in range(len(list)):
        if list[i] > max:
            max = list[i]
            argmax = i
    return argmax

@njit
def genetic_algorithm(generations, population_size, time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate, no_improvement_limit=3, mutation_rate=0.2):
    population = init_population(population_size)
    population_fitnesses = np.zeros(population_size)

    # calculate fitness for each individual in the population
    for i in range(population_size):
        population_fitnesses[i] = fitness(population[i], time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)

    best_fitness = np.max(population_fitnesses)
    best_individual = population[np.argmax(population_fitnesses)]
    no_improvement_count = 0

    for _ in range(generations):
        new_population = np.zeros((population_size, 2), dtype=np.int64)
        new_population_fitnesses = np.zeros(population_size)
        for idx in range(population_size // 2):
            # select two parents
            parent1 = random_choice_numba(population, population_fitnesses)
            parent2 = random_choice_numba(population, population_fitnesses)

            # perform crossover and mutation
            child1, child2 = crossover(parent1, parent2)
            children = [mutation(child1, mutation_rate), mutation(child2, mutation_rate)]
            
            # add children to new population
            for j, child in enumerate(children):
                new_population[2 * idx + j] = child
                new_population_fitnesses[2 * idx + j] = fitness(child, time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)
        
        best_index = np.argmax(new_population_fitnesses)
        current_best_fitness = new_population_fitnesses[best_index]
        current_best_individual = new_population[best_index]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= no_improvement_limit:
                break

        # prepare for next generation
        population = new_population
        population_fitnesses = new_population_fitnesses

    return best_individual



def random_search_optimizer(PARAMETER_ONE_LIMITS,PARAMETER_TWO_LIMITS,time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate):
    best_fitness = 0
    best_individual = None
    for _ in range(100):
        individual = np.zeros(2, dtype=np.int64)
        individual[0] = np.random.randint(PARAMETER_ONE_LIMITS[0], PARAMETER_ONE_LIMITS[1] + 1)
        individual[1] = np.random.randint(PARAMETER_TWO_LIMITS[0], PARAMETER_TWO_LIMITS[1] + 1)
        individual_fitness = fitness(individual, time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)
        if individual_fitness > best_fitness:
            best_fitness = individual_fitness
            best_individual = individual
    return best_individual

"""
try:
    best_individual = genetic_algorithm(GENERATIONS, POPULATION_SIZE, time, demand,lead_times,cost_per_transcation,hold_cost_per_unit,backlog_cost,sale_price,cost_price,tax_rate)

except:
    best_individual = random_search_optimizer(PARAMETER_ONE_LIMITS,PARAMETER_TWO_LIMITS,time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)
print(f" found results as  ROL = {best_individual[0]}, EOQ = {best_individual[1]}")
"""