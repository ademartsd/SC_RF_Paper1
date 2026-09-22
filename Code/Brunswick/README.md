################################################################
#                                                              #
#              MH-MCMC SEDIMENTARY ENVIRONMENT                 #
#                                                              #
#        Bayesian inversion of receiver functions (RFs)        # 
#        Output: MATRIX_res_{station}.csv                      #
#        By Ademar Fernández, September 2026                   # 
#        Contact: cfernandez@seoe.sc.edu                       #
#                                                              #
################################################################


################################################################
################### PART 1: INVERSION CODE #####################
################################################################

The inversion code is MH_MCMC_SED.py. Run it as:

(base) ademar@Z56A$ conda activate prs
(prs) ademar@Z56A$ python3 MH_MCMC_SED.py 

input: 1. Stream with RFs (stream_Z56A.h5)
	 2. RF stack (Z56A_trace_mean.txt)
	 3. RF STD (Z56A_trace_std.txt)

output: Once you run the code, the file MATRIX_res_Z56A.csv will be
	  created in the output folder. 

Please comment line 48 if you want to do the full run
(the full inversion takes about two months to run).

################################################################
################### PART 2: PLOTTING SCRIPT ####################
################################################################

The plotting script is PLOTS_SED.py. Run it as:

(base) ademar@Z56A$ conda activate prs
(prs) ademar@Z56A$ python3 PLOTS_SED.py

input: 1. Results from the inversion (./output/MATRIX_res_Z56A.csv)
	 2. Stream with RFs (stream_Z56A.h5)
	 3. RF stack (Z56A_trace_mean.txt)
	 4. RF STD (Z56A_trace_std.txt)

output: Once you run the code, the folder ./figs/ will be created, and the figures will be stored inside.

In line 5 (number_to_plot) you can modify the number of models to plot. Line 9 allows us to use all the traces (realistic). The realistic plot could take 24 hours; meanwhile, the fast view ones take just a couple of minutes.