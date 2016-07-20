
@ Author = Andrew Treadway
@ Class = CS 7641 -- Machine Learning
@ Date of Submission = Feb. 7, 2016

The code in this directory is written in R and the following information applies:

R Version Used: 3.12

R Packages Required:
			RWeka (for decision trees and boosting)
			neuralnet (for neural networks)
			nnet (for neural networks)
			knncat (for K-NN algorithm)
			class (for K-NN algorithm)
			ada (for boosting)
			e1071 (for support vector machines)
			caret (for feature selection and variable importance)
			pROC (for machine learning metrics)

			
The code consists of two R scripts:		Income Analysis.R
						Exercise.R
										
The scripts read in data from three different csv files (train and test set for the Salary Analysis script and a 
dataset for the Workout Analysis script).

It is assumed that the scripts are being run from inside the directory containing these csv files.

1) Income Analysis

Implements the machine learning algorithms on salary-related data.  The goal of this script
and analysis is to predict whether an individual makes above or below $50,000 based of various
factors.  The script is organized into sections, with each section covering one of the algorithms.

To best understand what the code is doing, each section should be run separately, as 
various variables defined in specific sections correspond to information given in 
specific sections of the analysis writeup.  
					
					
2) Workout Analysis
	
Implements the machine learning algorithms on workout-related data.  
The goals of this script and analysis are to predict whether an individual is performing 
a specific workout correctly based off various factors.  
The script, like the Income Analysis script, is organized into sections.  Similarly, it should be run section-by-section
to understand how the information from the code corresponds to that in the analysis writeup.  
