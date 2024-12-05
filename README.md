# CrysAlisPro_hklPlanes_processing

## Current Features
This code allows one to efficiently and automatically gather specific reciprocal space planes data files (in the case of CrysAlisPro, .img data files) from a cloud storage disc to his/hers local working directory creating and managing the local directory accordingly. After it processes all of the gathered data creating for each unique state (Temperature and Voltage in this case) a plot of the peak intensity of a certain number of hk points in reciprocal space along a range of l-values for the specified reciprocal space planes data files.

The code also provides one with a 'log.error' file where it writes all the encontered errors during the code execution and a 'log.data' file where it stores, for each unique state, relevant information like the peaks position, average peak intensity and the approximate charge order detected from the peak analysis.

## Contributions
This is a very infant and basic code still which means that bugs, errors, and surely improvements to the code with the objective of further automatizing the process, making it easier and more direct as well as less first-input dependent, are possible and more than welcome. There is also a possible, bigger goal of expanding the code's possible analysis tools and implementation within the analysis procedure for this type data which however much more ambitious is also very much welcome. Any contributions of any kind with the purpose of improving this type of analysis process are well recieved.
