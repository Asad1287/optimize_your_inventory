import matplotlib.pyplot as plt
import pandas as pd
import os 
SAVE_FOLDER = "/mnt/d/Portfolio/Inventory_Optimization/inventory_optimizer_app/gen_algo_simulation/single/plots"

def plot_inventory(on_hand, demand, in_transit):
    print("start prinitng")
    fig, axs = plt.subplots(3, figsize=(12, 15))
    

    axs[0].plot(on_hand)
    axs[0].set_title('On-Hand Inventory Over Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('On-hand inventory')


    axs[1].plot(demand)
    axs[1].set_title('Demand Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Demand')

    in_transit_total = in_transit.sum(axis=1)
    axs[2].plot(in_transit_total)
    axs[2].set_title('In-Transit Inventory Over Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('In-transit inventory')

  

    

    fig.tight_layout()
    #save figure to file
    fig.savefig(os.path.join(SAVE_FOLDER,"inventory.png"))

    fig.tight_layout()

    return fig

def print_results(results):
    df = pd.DataFrame(results)
    df.columns = ["cost_of_inventory",
        "cycle_service_level",
        "fill_rate", 
        "total_profit",
        "SL_alpha",
        "c_k",
        "c_h",
        "c_b",
    ]
    
    df.to_csv(os.path.join(SAVE_FOLDER,"results.csv"))

