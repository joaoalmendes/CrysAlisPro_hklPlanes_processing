# A script that gathers all the temperature and strain hkl mesh intensity informatio

## Gathers possibly to a dic first and then creates a kind of tuple matrix where it says which data is available
## For every point creates a coordinate matrix with the point intensity information and uses this for further analysis
## Following some criteria for the neiboughr phase space points (T, Strain), with the intensity matrixs, computes the difference between the two
## Now store (being ready to plot at any time) the new Difference matrix for a specific two points and stores this matrix in a dic (maybe) for further use
## The Difference Matrix information can then be used for plotting differences regarding temperature, strain, planes, or hkl point,... or even for further use
## The DM between neiboughring points can perhaps yield some interesting intensity changes for specific Bragg peaks with regard to temperature or strain changes
