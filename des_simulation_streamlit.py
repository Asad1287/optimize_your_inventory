import streamlit as st
from pydantic import BaseModel, Field
import numpy as np
from discrete_event_simulation.single.des_single_continous import *

class SimulationConfig(BaseModel):
    reorder_level:float = Field(..., example=10)
    reorder_qty:float = Field(..., example=20)
    purchase_cost:float = Field(..., example=1000)
    selling_price:float = Field(..., example=2000)
   
    holding_cost_unit : float = Field(..., example=2)
    ordering_cost : float = Field(..., example=1)
    other_costs : float = Field(..., example=4)
    lead_time_mean : float = Field(..., example=53)
    lead_time_std : float = Field(..., example=20)
    delivery_batches : float = Field(..., example=1)
    daily_demand_mean   : float = Field(..., example=0.0813)
    daily_demand_std : float = Field(..., example=12)
    review_period  : float = Field(..., example=1)
    backlog_cost_unit : float = Field(..., example=3)
    safety_stock  : float = Field(..., example=0)
    balance : float = Field(..., example=0)
    TIME : int = Field(..., example=1000)

class SimulationResult(BaseModel):
    cost_of_inventory : float
    cycle_service_level : float
    fill_rate : float
    total_profit : float

st.title("Welcome to the Inventory Optimizer App!")

reorder_level = st.number_input("Enter Reorder Level", value=10.0)
reorder_qty = st.number_input("Enter Reorder Quantity", value=20.0)
purchase_cost = st.number_input("Enter Purchase Cost", value=1000.0)
selling_price = st.number_input("Enter Selling Price", value=2000.0)
holding_cost_unit = st.number_input("Enter Holding Cost per Unit", value=2.0)
ordering_cost = st.number_input("Enter Ordering Cost", value=1.0)
other_costs = st.number_input("Enter Other Costs", value=4.0)
lead_time_mean = st.number_input("Enter Lead Time Mean", value=53.0)
lead_time_std = st.number_input("Enter Lead Time Standard Deviation", value=20.0)
delivery_batches = st.number_input("Enter Delivery Batches", value=1.0)
daily_demand_mean = st.number_input("Enter Daily Demand Mean", value=0.0813)
daily_demand_std = st.number_input("Enter Daily Demand Standard Deviation", value=12.0)
review_period = st.number_input("Enter Review Period", value=1.0)
backlog_cost_unit = st.number_input("Enter Backlog Cost per Unit", value=3.0)
safety_stock = st.number_input("Enter Safety Stock", value=0.0)
balance = st.number_input("Enter Balance", value=0.0)
TIME = st.number_input("Enter Time", value=1000, format="%d")

if st.button("Run Simulation"):
    config = SimulationConfig(
        reorder_level=reorder_level, 
        reorder_qty=reorder_qty, 
        purchase_cost=purchase_cost,
        selling_price=selling_price,
        holding_cost_unit=holding_cost_unit,
        ordering_cost=ordering_cost,
        other_costs=other_costs,
        lead_time_mean=lead_time_mean,
        lead_time_std=lead_time_std,
        delivery_batches=delivery_batches,
        daily_demand_mean=daily_demand_mean,
        daily_demand_std=daily_demand_std,
        review_period=review_period,
        backlog_cost_unit=backlog_cost_unit,
        safety_stock=safety_stock,
        balance=balance,
        TIME=TIME
    )

    cost_of_inventory, cycle_service_level, fill_rate, total_profit,plots = run_simulation_reorder(
        config.reorder_level, 
        config.reorder_qty, 
        config.purchase_cost, 
        config.selling_price, 
        config.holding_cost_unit, 
        config.ordering_cost, 
        config.other_costs, 
        config.lead_time_mean, 
        config.lead_time_std, 
        config.delivery_batches, 
        config.daily_demand_mean, 
        config.daily_demand_std, 
        config.review_period, 
        config.backlog_cost_unit, 
        config.safety_stock, 
        config.balance, 
        config.TIME,
        True
    )


  
    


    # Prepare the response
    result = SimulationResult(
        cost_of_inventory=cost_of_inventory,
        cycle_service_level=cycle_service_level,
        fill_rate=fill_rate,
        total_profit=total_profit
    )

    st.json(result.dict()) # Display the result as JSON
    st.pyplot(plots)