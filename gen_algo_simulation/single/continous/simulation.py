import sys
sys.path.append('/mnt/d/Portfolio/Inventory_Optimization/inventory_optimizer_app')
import numpy as np
import pandas as pd
from gen_algo_simulation.single.Demand import *
from gen_algo_simulation.single.Leadtime  import *
from gen_algo_simulation.plotter import *
import numpy as np
import pandas as pd
from numba import njit
from typing import List


@njit
def random_choice_numba(a, p):
    # a is array of choices
    # p is array of probabilities
    cumsum = np.cumsum(p)
    random_number = np.random.random()
    index = np.searchsorted(cumsum, random_number, side="right")
    return a[index]
  
@njit
def run_simulation(time, ROP, EOQ, transit, hand, unit_shorts, stockout_period, L_x, L_pmf, demand, 
                   cost_per_transcation, sale_price, cost_price, tax_rate, backlog_cost,physical_stock,hold_cost_per_unit):
    c_k = cost_per_transcation
    c_h = hold_cost_per_unit * demand[0]
    c_b = backlog_cost
    R = 1

    for t in range(time):
      
        hand[t] = hand[t-1] - demand[t] + transit[t-1,0]
       
        if t < time-1:
            transit[t+1,:-1] = transit[t,1:]      
            if t % R == 0:
                net = hand[t] + transit[t].sum()
                if net <= ROP:
                    actual_L = random_choice_numba(L_x, p=L_pmf)
                    transit[t+1,actual_L-1] = (1+ (ROP-net)//EOQ)*EOQ  

        if hand[t] > 0: 
            physical_stock[t] = (hand[t-1] + transit[t-1,0] + hand[t])/2
        else:
            physical_stock[t] = max(hand[t-1] + transit[t-1,0],0)**2/max(demand[t],1)/2


        if demand[t] == 0:
            unit_shorts[t] = 0
        
        elif hand[t] < 0:
            unit_shorts[t] = demand[t]
        
        elif demand[t]> hand[t] and hand[t] >= 0:
            unit_shorts[t] = demand[t] - hand[t]
        else:
            unit_shorts[t] = 0


        stockout_period[t] = hand[t] < demand[t]
      
      

        
        c_k = cost_per_transcation if hand[t] + transit[t].sum() <= ROP else 0
        
        c_h += (hand[t] + transit[t].sum()) * hold_cost_per_unit
        
        c_b += backlog_cost * unit_shorts[t]
       


        sales = min(hand[t], demand[t])
       
        revenue = sales * sale_price
        
        costs = sales * cost_price
        
        gross_profit = revenue - costs
       
        net_profit = gross_profit * (1 - tax_rate)
        total_profit = net_profit - (c_k + c_h)
        stockout_cycle = stockout_period.sum() 
        
        cost_of_inventory = (c_k + c_h + c_b)/time
        
        SL_alpha = 1 - stockout_cycle / time
        
        fill_rate = 1 - (unit_shorts.sum() / demand.sum())
        cycle_service_level = 1 - stockout_cycle / len(demand)
        over_stock = (hand[t] + transit[t].sum()) - demand[t]
        over_stock_percent = over_stock / demand.sum()


    return c_k, c_h,c_b, demand, hand, transit,total_profit,SL_alpha,fill_rate,cost_of_inventory,cycle_service_level

@njit
def simulation(ROP,EOQ , time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate):
    
    demand, demand_mu, demand_std, pmf = model_demand(demand,time)
    L_x, L_pmf, L_mu, L_std, L_median, L_max = get_leadtime_from_array(lead_times)
    
    hand = np.zeros(time)
    
    transit = np.zeros((time, L_max + 1))
    unit_shorts = np.zeros(time)
    stockout_period = np.full(time, False)
    physical_stock = np.zeros(time)
    

    

    return run_simulation(time, ROP, EOQ, transit, hand, unit_shorts, stockout_period, L_x, 
                          L_pmf, demand, cost_per_transcation, sale_price, 
                          cost_price, tax_rate,backlog_cost, physical_stock, hold_cost_per_unit)




@njit
def simulation_process_single_item(ROP:int,EOQ:int,time:int,demand:List[float],lead_times:List[float],cost_per_transcation:float,hold_cost_per_unit:float,backlog_cost:float,sale_price:float,cost_price:float,tax_rate:float):
    
    c_k, c_h,c_b, demand, hand, transit,total_profit,SL_alpha,fill_rate,cost_of_inventory,cycle_service_level = simulation(ROP,EOQ , time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)
    
    return cost_of_inventory, cycle_service_level, fill_rate, total_profit,demand,hand,transit




@njit
def simulation_process_single_item_all(ROP:int,EOQ:int,time:int,demand:List[float],lead_times:List[float],cost_per_transcation:float,hold_cost_per_unit:float,backlog_cost:float,sale_price:float,cost_price:float,tax_rate:float):
    
    c_k, c_h,c_b, demand, hand, transit,total_profit,SL_alpha,fill_rate,cost_of_inventory,cycle_service_level = simulation(ROP,EOQ , time, demand, lead_times, cost_per_transcation, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)
    
    return c_k, c_h,c_b, demand, hand, transit,total_profit,SL_alpha,fill_rate,cost_of_inventory,cycle_service_level


#print(demand)

"""
c_k, c_h, transit, hand, unit_shorts, stockout_period, physical_stock, total_profit,stockout_cycle,avg_cost = simulation(ROP,EOQ,time,item_prop1)

print("Total Profit:",total_profit)
print("Cost:",c_k + c_h)
print("SL_alpha:",1 - stockout_cycle / len(demand))
print("fill_rate:",1 - unit_shorts.sum() / demand.sum())
print("Stockout Cycle:",stockout_cycle)
print("Total Profit:",total_profit)
print("Unit shorts:",unit_shorts.sum())
print("Avg Cost:",avg_cost)


def print_cost_results(c_k,c_h,stockout_cycle,demand):

    cost = c_k + c_h
    SL_alpha = 1 - stockout_cycle / len(demand) # using simulated demand
    fill_rate = 1 - unit_shorts.sum() / demand.sum() # using simulated demand

    print("Total Profit:",total_profit)
    print("Cost:",cost)
    print("SL_alpha:",SL_alpha)
    print("fill_rate:",fill_rate)
    

plot_inventory(hand, demand, transit)

"""