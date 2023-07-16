## Inventory Optimization using Simulation Approach

This project is an initiative to develop an efficient, robust, and user-friendly inventory optimization tool using simulation techniques. It is designed to help businesses better understand their inventory dynamics, identify bottlenecks, and make more informed decisions about stock replenishment and resource allocation.

The project uses a combination of discrete-event simulation (DES) and analytical methods to create realistic scenarios for inventory management, considering factors like re-order points (ROP), economic order quantities (EOQ), demand patterns, lead times, and costs.

Under the hood, the project leverages the power of Python, Numba, Streamlit, and FastAPI to deliver a highly interactive, web-based interface for running simulations, as well as a FastAPI-powered backend for serving simulation results.

The entire simulation process is optimized with Numba, a just-in-time compiler for Python that translates a subset of Python and NumPy code into fast machine code. It's a game-changer for projects that require heavy numerical computations, providing significant speed improvements.

Key Features
Inventory Simulation: Run inventory simulations using various inputs such as ROP, EOQ, demand, lead times, and costs.
Discrete-Event Simulation: Generate realistic inventory scenarios using a DES approach.
Performance Optimization: Simulation code optimized with Numba for high-speed computations.
Interactive UI: An interactive web interface built with Streamlit for running simulations and visualizing the results.
API Integration: A FastAPI backend that serves the simulation results, allowing for seamless integration with other systems.
Installation & Usage
1. Firstly, clone this repository to your local machine using: git clone https://github.com/Asad1287/optimize_your_inventory.git
2. cd inventory-optimization
3. pip install -r requirements.txt
4. Run simulation for DES streamlit des_simulation_streamlit.py
5. Run optimization for inventory using FastAPI use des_simulation_api.py
6. Run optimization for invetory using genetic algorithm optimization use gen_single_item_opt_api.py
