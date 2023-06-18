from enum import Enum

class Setting(Enum):
    GENERATIONS = 100
    POPULATION_SIZE = 50
    MUTATION_RATE = 0.1
    TIME = 10000
    PARAMETER_ONE_LIMITS = (1,13)
    PARAMETER_TWO_LIMITS = (1,101)
