
@ Author = Andrew Treadway
@ Class = CS 7641 -- Machine Learning
@ Date of Submission = March 13, 2016

The code in this directory consists of a collection of scripts written in R, Python, and Java.  The code is organized according to the problem involved i.e. all of the code related to the
Knapsack probem is contained in one directory, labeled "Knapsack Code." The following information applies:

For the R scripts:
---------------------------------------------------------------------------------------------------

R Version Used: 3.12

R Packages Required:
						GenSA (for simualted annealing)
						GA (for genetic algorithms)
			

			
There are three different R scripts--one for each optimization problem.  Assuming you are in the root directory of the files, 
the scripts are as follows:
							Knapsack Problem/Knapsack Code.R
							4 Peaks Problem/4 Peaks code.R
							Hamming Weight Problem/hamming weight.R
							
										
		
Each of the R scripts implements genetic algorithms and simulated annealing for each of their repective optimization problems.
To best understand what the code is doing, the various sections of each script file should be run separately, as the variable names
and results from different sections correspond to sections in the analysis write-up.  		
										
---------------------------------------------------------------------------------------------------


For the Python scripts:
---------------------------------------------------------------------------------------------------

Python Version Used: 2.76

Python Packages Required:
							random (for random number generation -- used in hill climbing)
							time (for timing the clocktime of the algorithms)
							pandas (for reading in csv files)
							
The following additional packages were used for the MIMIC implementation:
							networkx
							matplotlib.pyplot
							numpy
							scipy

The Python MIMIC implementation is based off of the source code found here: https://github.com/mjs2600/mimicry/tree/jeff_ml_project_2							
			

			
There are five different Python scripts--two for each optimization problem, with the exception of the 4 Peaks problem, which has one associated script.  
Each problem's pair of scripts consists of one script for RHC and one for implementing the MIMIC algorithm.
Assuming you are in the root directory of the files, the scripts are as follows:

							Knapsack Problem/hill_climbing_knapsack.py
							Knapsack Problem/mimic_knapsack.py
							
							
							4 Peaks Problem/hill_climbing_4_peaks.py
							
							Hamming Weight Problem/hill_climbing_hamming_weight.py
							Hamming Weight Problem/mimic_hamming_weight.py
							
										
Each Python script implements the algorithm corresponding to its name (e.g. hill climbing) for its associated problem (e.g. knapsack).
										
---------------------------------------------------------------------------------------------------


For the Java Code:
---------------------------------------------------------------------------------------------------

JDK Version Used: 1.8

The Java code used the ABAGAIL library.  Two of the ABAGAIL files were modified for our analysis.  For the neural network training,
the Training Neural Network with Randomized Optimization.java and is located in Neural Network/AbaloneTest.java.  The AbaloneTest.java file was modified 
to suit our needs for implementing neural network models using three of the optimization algorithms to train the weights -- RHC, GA, and SA. 


The purpose of this code to train the weights of a neural network model using three different optimization algorithms.  These
are randomized hill climbing, genetic algorithms, and simualted annealing.  The neural network is a binary classifier that predicts
whether or not an individual makes greater than or less than $50K as an annual salary.  

The java file reads in data from a csv file, assumed to be in the current working directory of where the java program is run (change directory in java code if necessary).

Several varaiations of the parameters of the neural network model were attempted.  These different varaiations are included in the code, 
but commented out.  To see how they perform, one can uncomment out the respective line (see Java code file).  These varaiations include experimenting
with the number of hidden layers in the network, and the number of iterations used to train the weights of the network.

The data for this is located in Neural Network/income_data.txt

The second ABAGAIL file modified is the FourPeaksTest.java file.  This was only used for the MIMIC algorithm for the 4 Peaks Problem.  The other algorithms' implementations
were done in R and Python.

---------------------------------------------------------------------------------------------------


Each of the R and Python scripts pertaining to the Knapsack problem read in data from a csv file.  This data is contained in Knapsack Code/knapsack_data.csv
It is assumed that the scripts are being run from inside the directory containing these csv files.

	
					 