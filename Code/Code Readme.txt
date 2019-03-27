Non-negative tensor decomposition algorithm code:
	lraNTD.m
	call_tucker_als.m,
	call_tucker_als_opts.mat,
	fitness.m,
	scanparam.m,
	
Training and prediction code:
	PredictByMonopartite.m
	GenerateIdxForCV.m
	PredictS2_By_MatrixFactorization.m
	Measure_S2.m
	EstimationAUC.m
	prec_rec.m
	

Code example:
PredictByMonopartite(Tensor,Feature,CV,[P1 P2 P3],'Binary','S2');
Tensor : PK and PD data
Feature : drug feature
CV : Cross-validation(Usually set to 10)
P1 P2 P3:Tensor decomposition parameter
         and P1=P2，P3=2
		 P1 is drug latent feture