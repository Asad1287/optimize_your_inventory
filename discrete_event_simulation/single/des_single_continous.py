import numpy as np
import simpy
import matplotlib.pyplot as plt
np.random.seed(0)
from math import ceil

from typing import List, Tuple, Callable
from discrete_event_simulation.single.simulation_classes import *


SAVE_PLOTS_FOLDER = "./discrete_event_simulation/plots"
class Inventory_Simulation:

    def __init__(self, env:simpy.Environment, constants: Constants, variables: Variables, functions: ExternalFunctions):
        self.constants = constants
        self.variables = variables
        self.functions = functions
        self.env = env
    def generate_interarrival(self) -> float:
        return self.functions.generate_interarrival(self.constants.customer_arrival)
    
    def generate_leadtime(self) -> float:
        __leadtime = self.functions.generate_leadtime(self.constants.lead_time_mean, self.constants.lead_time_std, 1).item()
        return(__leadtime if __leadtime>=0 else self.constants.lead_time_mean)
    def customer_generate_demand(self) -> float:
        return self.functions.generate_demand(self.constants.customer_purchase_mean, self.constants.customer_purchase_std, 1).item()

    def generate_demand(self) -> float:
        __temp =  self.functions.generate_demand(self.constants.daily_demand_mean, self.constants.daily_demand_std, 1).item()
        return(__temp if __temp>=0 else 0)

    def total_cost(self):
        """
        Calculate the total ordering cost.
        """
        return self.variables.holding_cost+ self.variables.num_orders_placed * self.constants.ordering_cost

    def calculate_profit(self) -> float:
        """
        Calculate the total profit.
        """
        total_revenue = self.constants.selling_price * sum(self.variables.demands)
        total_unmet_demand_cost = self.constants.selling_price * self.variables.service_outage_cnt
        total_cost = self.calculate_total_cost() + total_unmet_demand_cost
        tax_rate = 0.2
        tax = tax_rate * total_revenue

        profit = total_revenue - total_cost - tax
        return profit


    def handle_order(self):
        # adjust reorder amount to cover negative inventory (backlogged orders)
        required_inventory = max(self.constants.reorder_level + 1, -self.variables.inventory) 

        self.variables.num_ordered = required_inventory
        self.variables.num_ordered = ceil(self.variables.num_ordered/self.constants.reorder_qty)*self.constants.reorder_qty

        self.variables.balance = self.constants.purchase_cost*self.variables.num_ordered + self.constants.ordering_cost
        self.variables.num_orders_placed += 1
        for _ in range(int(1/self.constants.delivery_batches)):
            yield self.env.timeout(self.generate_leadtime())
            self.variables.inventory += self.variables.num_ordered*self.constants.delivery_batches
        self.variables.num_ordered = 0

    
    def observe(self):
        while True:
            self.variables.obs_time.append(self.env.now)
            self.variables.inventory_level.append(self.variables.inventory)
            if self.variables.demand > self.variables.inventory: 
                self.variables.service_outage_cnt += 1
            self.variables.costslevel.append(self.variables.balance)
            yield self.env.timeout(1)
    def runner_setup(self):
        while True:
            interarrival = 1
            yield self.env.timeout(interarrival)
            self.variables.holding_cost.append(max(self.variables.inventory * self.constants.holding_cost_unit, 0)) # holding cost only applies to positive inventory

            self.variables.demand = self.generate_demand()
            self.variables.demands.append(self.variables.demand)
            self.variables.balance += self.constants.selling_price * min(self.variables.demand, self.variables.inventory)
            self.variables.inventory -= self.variables.demand

            #if self.variables.demand < self.variables.inventory:
            #    self.variables.balance += self.constants.selling_price*self.variables.demand
            #    self.variables.inventory -= self.variables.demand
            #else:
            #    self.variables.balance += self.constants.selling_price*self.variables.inventory
            #    self.variables.inventory = 0 
            if self.variables.inventory <= (self.constants.reorder_level + self.variables.safety_stock) and self.variables.num_ordered ==0:
                self.env.process(self.handle_order())

    def plot(self, plot_type):
        plt.figure()
        if plot_type == 'inventory':
            plt.step(self.variables.obs_time, self.variables.inventory_level)
            plt.ylabel('SKU level')
        elif plot_type == 'balance':
            plt.step(self.variables.obs_time, self.variables.costslevel)
            plt.ylabel('SKU balance USD')
        elif plot_type == 'sales':
            plt.step(self.variables.obs_time, self.variables.costslevel)
            plt.ylabel('SKU balance USD')
        elif plot_type == 'holding':
            plt.step(np.arange(len(self.variables.holding_cost)), self.variables.holding_cost)
            plt.ylabel('holding costs')
        elif plot_type == 'demand':
            plt.step(np.arange(len(self.variables.demands)), self.variables.demands)
            plt.ylabel('Demands')
        else:
            raise ValueError(f'Invalid plot_type: {plot_type}')
        plt.xlabel('Time')
        plt.savefig(f'{SAVE_PLOTS_FOLDER}/{plot_type}.png')
        return plt

    def plot_all_inventory(self):
    
        fig, axs = plt.subplots(3, figsize=(12, 15))
        

        #plot inventory level   
        axs[0].step(self.variables.obs_time, self.variables.inventory_level)
        axs[0].set_ylabel('SKU level')
        axs[0].set_xlabel('Time')
        axs[0].set_title('Inventory Level')

        #plot balance
        axs[1].step(self.variables.obs_time, self.variables.costslevel)
        axs[1].set_ylabel('SKU balance USD')
        axs[1].set_xlabel('Time')
        axs[1].set_title('Balance')
        
        #plot holding cost
        axs[2].step(np.arange(len(self.variables.holding_cost)), self.variables.holding_cost)
        axs[2].set_ylabel('holding costs')
        axs[2].set_xlabel('Time')
        axs[2].set_title('Holding Cost')
        fig.tight_layout()

        return fig

  

    





    def service_level(self):
        __temp_level= np.array(self.variables.inventory_level)
        __temp_level1 = __temp_level[__temp_level==0]
        if len(__temp_level1)==0:
            return 1
        else:
            return (1 - len(__temp_level1)/len(__temp_level))

    def avg_cost_of_inventory(self):
        #calculate the average cost of inventory holding
        return sum(self.variables.holding_cost)/len(self.variables.holding_cost)

    def print_service_level(self):
        return 1-self.variables.service_outage_cnt

    def total_holding_cost(self):
        """
        Calculate the total holding cost.
        """
        return sum(self.variables.holding_cost)
        
    def calculate_fill_rate(self):
        """
        Calculate the fill rate.
        """
        total_demand = sum(self.variables.demands)
        unmet_demand = self.variables.service_outage_cnt
        if total_demand == 0: 
            return 1
        return (total_demand - unmet_demand) / total_demand

    def calculate_cycle_rate(self):
        # calculate the cycle service level
        stockout_cycles = [1 for level in self.variables.inventory_level if level < 0]
        return 1 - sum(stockout_cycles) / len(self.variables.inventory_level)
      
    def service_level_alpha(self):
        stockout_cycles = [1 for level in self.variables.inventory_level if level < 0]
        return 1 - sum(stockout_cycles) / len(self.variables.inventory_level)

    def calculate_fill_rate(self):
        stockout_periods = [1 for demand, level in zip(self.variables.demands, self.variables.inventory_level) if demand > level]
        return 1 - sum(stockout_periods) / len(self.variables.demands)

    def calculate_total_cost(self):
        total_holding_cost = sum(self.variables.holding_cost)
        total_ordering_cost = self.variables.num_orders_placed * self.constants.ordering_cost
        return total_holding_cost + total_ordering_cost + self.variables.backlog_cost


def run(simulation:Inventory_Simulation,TIME):
    simulation.env.process(simulation.runner_setup())
    simulation.env.process(simulation.observe())
    simulation.env.run(until=TIME)

constants = Constants(500, 2000, 1000, 2000, 10, 20, 2)
variables = Variables(constants)
functions = ExternalFunctions()

env = simpy.Environment()
inventory_simulation = Inventory_Simulation(env, constants, variables, functions)


run(inventory_simulation, 10000)

print(f"Service Level: {inventory_simulation.service_level()}")
print(f"Total Holding Cost: {inventory_simulation.total_holding_cost()}")
print(f"Fill Rate: {inventory_simulation.calculate_fill_rate()}")
print(f"Cycle Rate: {inventory_simulation.calculate_cycle_rate()}")
print(f"Service Level (Alpha): {inventory_simulation.service_level_alpha()}")
print(f"Total Cost: {inventory_simulation.calculate_total_cost()}")
print(f"Fill Rate: {inventory_simulation.calculate_fill_rate()}")
print(f"Cycle Rate: {inventory_simulation.calculate_cycle_rate()}")


inventory_simulation.plot('inventory')
inventory_simulation.plot('balance')
inventory_simulation.plot('holding')


def run_simulation_reorder(reorder_level: int, reorder_qty: int, purchase_cost: float, selling_price: float, holding_cost_unit: float, ordering_cost: float, other_costs: float, lead_time_mean: float, lead_time_std: float, delivery_batches: int, daily_demand_mean: float, daily_demand_std: float, review_period: int, backlog_cost_unit: float, safety_stock: int, balance: int, TIME: int,return_plot:bool=False) -> Tuple[float, float, float, float]:

    
    
 

    constants = Constants(reorder_level, reorder_qty, purchase_cost, selling_price, holding_cost_unit, ordering_cost, other_costs, lead_time_mean, lead_time_std, delivery_batches, daily_demand_mean, daily_demand_std, review_period, backlog_cost_unit, safety_stock, balance)
    variables = Variables(constants)
    functions = ExternalFunctions()

    env = simpy.Environment()
    inventory_simulation = Inventory_Simulation(env, constants, variables, functions)
    run(inventory_simulation, TIME)

    cost_of_inventory = inventory_simulation.avg_cost_of_inventory()
    cycle_service_level = inventory_simulation.calculate_cycle_rate()
    fill_rate = inventory_simulation.calculate_fill_rate()
    total_profit = inventory_simulation.calculate_profit()
    plots = inventory_simulation.plot_all_inventory()

    if return_plot:
        return  cost_of_inventory, cycle_service_level, fill_rate, total_profit, plots
    else:
        return  cost_of_inventory, cycle_service_level, fill_rate, total_profit