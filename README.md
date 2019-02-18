# fmri_ISC_tools

This repo contains three functions to perform fmri ISC analyses:

bramila_ISC.m (+ bramila_ISC_worker.m)
- Run ISC analysis with "one-vs-rest" approach, no pairwise comparisons. Takes nifti files as input. Does one group and two groups analyses. Can only do cluster computations!

bramila_mantel_ISC.m
- Run ISC analysis for pair-wise ISC correlations matrix. Takes ISC matrix as input and compares it against a model matrix. Runs locally with parallel workers.

bramila_ttest2_ISC.m
- Run ISC analysis for pair-wise ISC correlation matrix. Takes ISC matrix as input and compares two groups using t-test2 type analysis. Runs locally.

bramila_supervised_ISC.m
- Run ISC analysis with specific subject-wise targets and simple k-nearest neighbors learning algorithm. Both regression and (binary) classification are supported. Runs locally.

18.2.2019 Janne Kauttonen