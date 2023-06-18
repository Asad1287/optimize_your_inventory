from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np

from gen_algo_simulation.simulation_settings import *

from gen_algo_simulation.single.continous.simulation import *


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

# Define the app
app = FastAPI()

# Define the request body model
class SimulationConfig(BaseModel):
    
    ROP: int = Field(..., example=10)
    EOQ: int = Field(..., example=100)
    TIME: int = Field(..., example=100)
    demand: list = Field(..., example=[10,20,30,40,50,60,70,80,90,100])
    lead_times: list = Field(..., example=[1,2,3,4,5,6,7,8,9,10])
    cost_per_transcation:float = Field(..., example=1)
    hold_cost_per_unit:float = Field(..., example=1)
    backlog_cost:float = Field(..., example=1)
    sale_price:float = Field(..., example=10)
    cost_price:float = Field(..., example=5)
    tax_rate:float = Field(..., example=0.2)


    

class SimulationResult(BaseModel):
    cost_of_inventory : float
    cycle_service_level : float
    fill_rate : float
    total_profit : float

@app.get("/")
async def root():
    return {"message": "Welcome to the Fast Inventory Optimizer API!"}


@app.post("/run_simulation", response_model=SimulationResult)
async def run_simulation(config: SimulationConfig):
    # Unpack the simulation parameters from the request body
    
    ROP = config.ROP
    EOQ = config.EOQ
    TIME = config.TIME
    demand = config.demand
    lead_times = config.lead_times
    cost_per_transaction = config.cost_per_transcation
    hold_cost_per_unit = config.hold_cost_per_unit
    backlog_cost = config.backlog_cost
    sale_price = config.sale_price
    cost_price = config.cost_price
    tax_rate = config.tax_rate
    
    cost_of_inventory, cycle_service_level, fill_rate, total_profit,_,_,_ = simulation_process_single_item(ROP,EOQ,TIME,demand,lead_times,cost_per_transaction,hold_cost_per_unit,backlog_cost,sale_price,cost_price,tax_rate)
    # Run the simulation
    
    # Prepare the response
    response = SimulationResult(
        cost_of_inventory=cost_of_inventory,
        cycle_service_level=cycle_service_level,
        fill_rate=fill_rate,
        total_profit=total_profit
    )

    # Return the response as JSON
    return response
