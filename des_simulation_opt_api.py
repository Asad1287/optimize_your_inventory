from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from discrete_event_simulation.single.simulation_classes import *
from discrete_event_simulation.single.des_single_continous import *
from discrete_event_simulation.single.des_random_optimzer import *
# Define the app
app = FastAPI()

# Define the request body model
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
    ROP : float
    EOQ : float
    TotalCost : float


@app.get("/")
async def root():
    return {"message": "Welcome to the Fast Inventory Optimizer API!"}


@app.post("/run_simulation", response_model=SimulationResult)
async def run_simulation(config: SimulationConfig):
    
    reorder_level = config.reorder_level
    reorder_qty = config.reorder_qty
    purchase_cost = config.purchase_cost
    selling_price = config.selling_price
   
    holding_cost_unit = config.holding_cost_unit
    ordering_cost = config.ordering_cost
    other_costs = config.other_costs
    lead_time_mean = config.lead_time_mean
    lead_time_std = config.lead_time_std
    delivery_batches = config.delivery_batches
    daily_demand_mean  = config.daily_demand_mean
    daily_demand_std : float = Field(..., example=12)
    review_period  : float = Field(..., example=1)
    backlog_cost_unit : float = Field(..., example=3)
    safety_stock  : float = Field(..., example=0)
    balance : float = Field(..., example=0)
    TIME : int = Field(..., example=1000)

    Setting = Setting(reorder_level,reorder_qty,purchase_cost,selling_price,holding_cost_unit,ordering_cost,other_costs,lead_time_mean,lead_time_std,delivery_batches,daily_demand_mean,daily_demand_std,review_period,backlog_cost_unit,safety_stock,balance,TIME)    
    
    best_ROP, best_EOQ, best_cost = simulated_annealing_optimizer(1000, Setting)
    # Run the simulation
    
    # Prepare the response
    response = SimulationResult(
       ROP = best_ROP,
       EOQ = best_EOQ,
       TotalCost = best_cost
    )

    # Return the response as JSON
    return response
