function [AUC_S2,AUPR_S2] = Measure_S2(Predicted_Scores,trn_Adj,tst_Adj, RemoveAllZeroTrn)
%% global measure
if nargin <4
    RemoveAllZeroTrn = false;
end
if RemoveAllZeroTrn
    threshold  =1; disp('only S2/S3');
else
    threshold  =0;
end


degrees_= sum(trn_Adj,1); % to remove S4 cases, if occured.
Scores  = Predicted_Scores(:,degrees_>=threshold);
Label_mat = tst_Adj(:,degrees_>=threshold);

TrueScore_PostiveClass =  Scores( Label_mat(:)>0) ; % LOG:change it to adapt -1,0,+1 triple-class
FalseScore= Scores( Label_mat(:)==0);
TrueScore_NegativeClass =  - Scores( Label_mat(:)<0) ; % LOG:change it to adapt -1,0,+1 triple-class

[AUC_S2, AUPR_S2 ]=EstimationAUC([TrueScore_PostiveClass;TrueScore_NegativeClass ],FalseScore,2000,0);