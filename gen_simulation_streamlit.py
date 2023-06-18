import streamlit as st
import numpy as np

from gen_algo_simulation.simulation_settings import *
from gen_algo_simulation.single.continous.simulation import *
from gen_algo_simulation.plotter import *


GENERATIONS = Setting.GENERATIONS.value
POPULATION_SIZE = Setting.POPULATION_SIZE.value
MUTATION_RATE = Setting.MUTATION_RATE.value
TIME = Setting.TIME.value
PARAMETER_ONE_LIMITS = Setting.PARAMETER_ONE_LIMITS.value
PARAMETER_TWO_LIMITS = Setting.PARAMETER_TWO_LIMITS.value


# Simulation input fields
ROP = st.number_input('ROP', value=10)
EOQ = st.number_input('EOQ', value=100)
TIME = st.number_input('TIME', value=100)
demand = st.text_input('Demand (comma-separated)', '10,20,30,40,50,60,70,80,90,100')
lead_times = st.text_input('Lead times (comma-separated)', '1,2,3,4,5,6,7,8,9,10')
cost_per_transaction = st.number_input('Cost per transaction', value=1.0)
hold_cost_per_unit = st.number_input('Hold cost per unit', value=1.0)
backlog_cost = st.number_input('Backlog cost', value=1.0)
sale_price = st.number_input('Sale price', value=10.0)
cost_price = st.number_input('Cost price', value=5.0)
tax_rate = st.number_input('Tax rate', value=0.2)

# Convert string inputs to list
demand = list(map(int, demand.split(',')))
lead_times = list(map(int, lead_times.split(',')))

if st.button('Run Simulation'):
    cost_of_inventory, cycle_service_level, fill_rate, total_profit,demand,hand,transit = simulation_process_single_item(ROP, EOQ, TIME, demand, lead_times, cost_per_transaction, hold_cost_per_unit, backlog_cost, sale_price, cost_price, tax_rate)
    
    # Display the results
    st.write(f'Cost of Inventory: {cost_of_inventory}')
    st.write(f'Cycle Service Level: {cycle_service_level}')
    st.write(f'Fill Rate: {fill_rate}')
    st.write(f'Total Profit: {total_profit}')

    
    plot_object = plot_inventory(hand, demand, transit)
    st.pyplot(plot_object)


