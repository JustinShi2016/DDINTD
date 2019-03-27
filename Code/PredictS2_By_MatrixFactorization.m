function Predicted_LabelMat  = PredictS2_By_MatrixFactorization(trn_DDI_mat,TrnFeatureMat,tst_DDI_mat, TstFeatureMat,  option_)
% 
nComp = option_;

Predicted_LabelMat = PLSregression_S2(trn_DDI_mat,TrnFeatureMat,TstFeatureMat,nComp, false);

%%
function Row_adj_predicted= PLSregression_S2(TrnAdj_S2,Fd_trn,Fd_tst,nComp, show)
if nargin < 5
    show =true;
end

nT = size(TrnAdj_S2,1);
Slice_one = TrnAdj_S2(:,1:nT);
Slice_two = TrnAdj_S2(:,nT+1:end);

Trn_Adj(:,:,1) = Slice_one;
Trn_Adj(:,:,2) = Slice_two;
%% ntf Non-negative Tucker Factorization
Trn_Adj = tensor(Trn_Adj);
opts=struct('NumOfComp',nComp,'FacAlg','als','MaxIter',300,'MaxInIter',20,...
            'TDAlgFile','call_tucker_als_opts.mat');
[Ydec]=lraNTD(Trn_Adj,opts);

fprintf('Complete. Fit=%f\n',fitness(Trn_Adj,Ydec));
core_Adj = Ydec.core;
PK_H = Ydec.U{1};
PD_H = Ydec.U{2};
Base = Ydec.U{3};

%% PLS
threshold1 = nComp(1,1);
PK_tst_predicted = DoPLS(Fd_trn,Fd_tst,PK_H,threshold1);   

%% Turker张量分解乘法
A{1} = PK_tst_predicted;
A{2} = PD_H;
A{3} = Base;
Row_adj_predicted = ttm(tensor(core_Adj),A);
Row_adj_predicted = double(tenmat(Row_adj_predicted,1));
% Row_adj_predicted  =PK_tst_predicted * Col_Response_trn' ;% since RowView_Response_trn already contains sqrt(S_adj)


%% core funtion
function [Y_tst_pred, Beta_ ] = DoPLS(X_trn,X_tst,Y_trn,LV)
[~,~,~,~,Beta_]=plsregress( X_trn,Y_trn ,LV );
Y_tst_pred = ([ones(size(X_tst,1),1)  ,  X_tst]*Beta_);
