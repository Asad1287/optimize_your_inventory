import streamlit as st

import numpy as np
from simulation_settings import *

from plotter import *

from Leadtime import *

from Demand import model_demand

from numba import njit
from gen_algo_simulation.single.continous.simulation import *


import pandas as pd

def run_app():

    st.title("Inventory Simulation")

    # Inputs from user
    demand = st.text_input("Enter demand (comma-separated numbers): ")
    demand = [int(i) for i in demand.split(",")]

    lead_time = st.text_input("Enter lead time (comma-separated numbers): ")
    lead_time = [int(i) for i in lead_time.split(",")]

    ROP = st.number_input("Enter ROP: ", min_value=0, step=1)
    EOQ = st.number_input("Enter EOQ: ", min_value=0, step=1)
    TIME = st.number_input("Enter TIME: ", min_value=0, step=1)

    cost_per_transaction = st.number_input("Enter cost per transaction: ", min_value=0.0, step=0.01)
    hold_cost_per_unit = st.number_input("Enter holding cost per unit: ", min_value=0.0, step=0.01)
    backlog_cost = st.number_input("Enter backlog cost: ", min_value=0.0, step=0.01)
    sale_price = st.number_input("Enter sale price: ", min_value=0.0, step=0.01)
    cost_price = st.number_input("Enter cost price: ", min_value=0.0, step=0.01)
    tax_rate = st.number_input("Enter tax rate: ", min_value=0.0, step=0.01)

    # Once all inputs are given, we run the simulation
    if st.button("Run Simulation"):
        demand, demand_mu, demand_std, pmf = model_demand(demand,TIME)
        L_x, L_pmf, L_mu, L_std, L_median, L_max = get_leadtime_from_array(lead_time)

        

        c_k, c_h,c_b, demand, hand, transit,total_profit,SL_alpha,fill_rate,cost_of_inventory,cycle_service_level = simulation(ROP,EOQ,TIME,demand,lead_time,cost_per_transaction,hold_cost_per_unit,backlog_cost,sale_price,cost_price,tax_rate)


        st.write(f"SL_alpha: {SL_alpha}")
        st.write(f"fill_rate: {fill_rate}")
        st.write(f"Total Profit: {total_profit}")
        st.write(f"Cost of Inventory: {cost_of_inventory}")
        st.write(f"Cycle Service Level: {cycle_service_level}")

        fig_inventory = plot_inventory(hand, demand, transit)
        st.pyplot(fig_inventory)

        

if __name__ == "__main__":
    run_app()