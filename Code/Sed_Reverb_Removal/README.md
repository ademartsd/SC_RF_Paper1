################################################################
#                                                              #
#              SEDIMENT REVERBERATION FILTER	               #
#                                                              #
#        Code to remove reverb. from receiver functions (RFs)  # 
#        Input: raw stream with RFs (reverberating)            #
#        Output: stream of curated RFs                         #
#        By Dr. Shubham Agrawal                                # 
#        Contact: SAGRAWAL@mailbox.sc.edu ,                    #
#	          or contact me at cfernandez@seoe.sc.edu      #
#                                                              #
################################################################

################################################################
If you use it, please cite (together with this work):

Agrawal, S., Eakin, C.M., O’Donnell, J.P., 2023. Tracking crustal thick-
ness at the sediment-inundated edge of the gawler craton, south australia.
Tectonophysics 862, 229938.

################################################################

The code is reverb_removal.py. Run it as:

(base) ademar@Z56A$ conda activate prs
(prs) ademar@Z56A$ python3 reverb_removal.py

input:   1. Stream with raw RFs (stream_Z56A_RAW_STILL_REVERB.h5)

output: Once you run the code, the curated stream (stream_Z56A.h5)
 	will be created in the output folder. This file will be the
	output for the inversion for station Z56A. Both are the same
	files.