
@ Author = Andrew Treadway
@ GT ID = atreadway6
@ Class = CS 7641 -- Machine Learning
@ Date of Submission = April 3, 2016

The code in this directory is written in R and the following information applies:

R Version Used: 3.12

R Packages Required:

			stats (for k-means clustering and PCA)
			fpc (for clustering metrics)
			mclust (for EM clustering)
			factoextra (for PCA plotting)
			ica (independent components analysis)
			moments (for calculating kurtosis for ICA)
			neuralnet (for neural nets)
			pROC (for roc curves and AUC metrics)
			randomProjection (for Random Projection)
			FSelector (provides information gain functionality for feature selection)
			plyr (for utility functions)

			
The code consists of two R scripts:		Income Analysis -- Unsupervised.R
										Exercise -- Unsupervised.R
										
The scripts read in data from three different csv files (a data for the Salary Analysis script and a dataset for the Workout Analysis script).
It is assumed that the scripts are being run from inside the directory containing these csv files.

1) Income Analysis
					Implements the machine learning algorithms on salary-related data.  The goal of this script
					and analysis is to use unsupervised learning algorithms to analyze the attributes and structure of income-related data associated 
					with individuals. The labels associated with the data points classify whether or not an indivudal makes above or below $50,000. 
					The script is organized into sections, with each section covering one of the algorithms.
					
					To best understand what the code is doing, each section should be run separately, as various variables
					defined in specific sections correspond to information given in specific sections of the analysis writeup. 

					This script has two corresponding csv data files -- train_income.csv and test_income.csv
					
					
2) Workout Analysis
	
					Implements the machine learning algorithms on workout-related data.  The goal of this script
					and analysis is to use unsupervised learning algorithms to analyze the attributes and structure of workout-related data, with labels
					classifying what workout movement an individual performed.  
					The script, like the Income Analysis script, is organized into sections.  Similarly, it should be run section-by-section
					to understand how the information from the code corresponds to that in the analysis writeup.  
					
					This has a corresponding csv data file -- weight_lifting_data.csv.