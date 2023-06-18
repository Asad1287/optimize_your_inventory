import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from typing import List, Callable
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Constants:
    
    reorder_level: Optional[int] = 10
    reorder_qty: Optional[int] = 20
    purchase_cost: Optional[int] = 1000
    selling_price: Optional[int] = 2000
   
    holding_cost_unit: Optional[int] = 2
    ordering_cost: Optional[int] = 1
    other_costs: Optional[int] = 4
    lead_time_mean: Optional[int] = 53
    lead_time_std: Optional[int] = 20
    delivery_batches: Optional[int] = 1
    daily_demand_mean: Optional[int] = 0.0813
    daily_demand_std: Optional[int] = 12
    review_period : Optional[int] = 1
    backlog_cost_unit : Optional[int] = 3
    safety_stock : Optional[int] = 0
    balance : Optional[int] = 0
    
@dataclass
class Variables:
    constants : Constants = Constants(10, 20, 1000, 2000, 10, 20, 2)
    safety_stock: int = constants.safety_stock
    balance: float = constants.balance
    num_ordered: int = 0
    inventory: float = field(default_factory=lambda: np.random.uniform(0, constants.reorder_qty, 1).item())
    obs_time: List[float] = field(default_factory=list)
    inventory_level: List[float] = field(default_factory=list)
    demand: float = constants.daily_demand_mean
    costslevel: List[float] = field(default_factory=list)
    saleslevel: List[float] = field(default_factory=list)
    holding_cost: List[float] = field(default_factory=list)
    service_outage_cnt: int = 0
    demands: List[float] = field(default_factory=list)
    num_orders_placed : int = 0 # number of orders placed
    backlog_cost : int = constants.backlog_cost_unit
    
    def __post_init__(self):
        self.safety_stock = int(0.2*constants.reorder_level)





@dataclass
class Setting:
    reorder_level : float = 10
    reorder_qty : float = 20
    purchase_cost : float = 1000
    selling_price : float = 2000
   
    holding_cost_unit : float = 2
    ordering_cost : float = 1
    other_costs : float = 4
    lead_time_mean : float = 53
    lead_time_std : float = 20
    delivery_batches : float = 1
    daily_demand_mean : float = 0.0813
    daily_demand_std : float = 12
    review_period  : float = 1
    backlog_cost_unit : float = 3
    safety_stock  : float = 0
    balance  : float = 0
    TIME : int = 1000



@dataclass
class ExternalFunctions:
    generate_interarrival: Callable = np.random.exponential
    generate_leadtime: Callable = np.random.normal
    generate_demand: Callable = np.random.normal

constants = Constants(10, 20, 1000, 2000, 10, 20, 2)
variables = Variables(constants)
functions = ExternalFunctions()
