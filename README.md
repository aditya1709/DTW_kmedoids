# DTW_kmedoids
Multivariate time series clustering using Dynamic Time Warping (DTW) and k-mediods algorithm
This repository contains code for clustering of multivariate time series using DTW and k-mediods algorithm. It contains code for optional use of LB_Keogh method for large data sets that reduces to linear complexity compared to quadratic complexity of dtw.
The train data should be a numpy array of the form (M,N,D) where
1. M - Number of data sequences.
2. N - length of data sequences.
3. D - Dimension of data sequences (number of features).
