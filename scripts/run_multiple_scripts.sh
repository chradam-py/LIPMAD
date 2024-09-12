#!/bin/bash

# Execute the first program
echo "Starting program1..."
python 2a_detect_sources.py -f ../../data/ANF_TopDown/iso400_t_1_100s/2023-10-
echo "program1 finished."
# Execute the first program

echo "Starting program2..."
python 2b_measure_photometry.py ../../data/ANF_TopDown/iso400_t_1_100s/2023-10-
echo "program2 finished."
# one and two finished on 14052024

# Execute the first program
echo "Starting program3..."
python 2c_analyse_data.py ../../data/ANF_TopDown/iso400_t_1_100s/2023-10-
echo "program3 finished."

# # Execute the second program
# echo "Starting program4..."
# python 3a_georef_data.py ../../data/ANF_TopDown/iso400_t_1_100s/2023-10-
# echo "program4 finished."

# Execute the first program
echo "Starting program5..."
python 3b_plot_data.py ../../data/ANF_TopDown/iso400_t_1_100s/2023-10-
echo "program5 finished."
