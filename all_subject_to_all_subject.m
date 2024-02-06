clear all
close all
clc
tic
fname=mfilename('fullpath');
[pathstr,~,~]=fileparts(fname)
source_data = 1
% 1:Sc sub1~39
% 2:St sub1~22
% 3:DREAM sub1~20
% 4:MASS_SS1 sub1~47
target_data = 2
% 1:Sc sub1~39
% 2:St sub1~22
% 3:DREAM sub1~20
% 4:MASS_SS1 sub1~47
transfer_learning = 1
% 0:CCA
% 1:subspace align
% 2:SSTCA
% 3:Reduced_TCA
% 4:DICD 
boundary = 0
% 1: avoiding edge effects by considering a much longer signal temporarily 
% 0: the edge artifacts may contaminate the part of the feature we are interested in.
time_freq_method = 'scattering' ;
% RCWT or scattering

plot_name = 'DREAM - MASS SS1 SA normalized confusion matrix (SVM)';

%% Choose the source data
2D_plot = 0
if (source_data == 1)
    %% Import SC
    
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])
    Nsubs = 39; % numer of subjects
    
    X1 = readmatrix([pathstr '\data\PSG\SC_FPzCz_scattering.txt']);
    Y1 = readmatrix([pathstr '\data\PSG\SC_PzOz_scattering.txt']);
    t1 = readmatrix([pathstr '\data\Stage\SC_stage.txt']);
    epoch_num1 = readmatrix([pathstr '\data\Stage\SC_epoch_number.txt']);
elseif (source_data ==2)
    %% Import ST  
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])

    Nsubs = 22; % numer of subjects
    X1 = readmatrix([pathstr '\data\PSG\ST_FPzCz_scattering.txt']);
    Y1 = readmatrix([pathstr '\data\PSG\ST_PzOz_scattering.txt']);
    t1 = readmatrix([pathstr '\data\Stage\ST_stage.txt']);
    epoch_num1 = readmatrix([pathstr '\data\Stage\ST_epoch_number.txt']);
elseif (source_data ==3)
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])

    Nsubs = 20; % numer of subjects
    X1 = readmatrix([pathstr '\data\PSG\DREAM_O1A2_scattering.txt']);
    Y1 = readmatrix([pathstr '\data\PSG\DREAM_O2A1_scattering.txt']);
    t1 = readmatrix([pathstr '\data\Stage\DREAM_stage.txt']);
    epoch_num1 = readmatrix([pathstr '\data\Stage\DREAM_epoch_number.txt']);

    elseif (source_data == 4)
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])

    Nsubs = 47; % numer of subjects
    X1 = readmatrix([pathstr '\data\PSG\MASS_SS1_O1CLE_scattering.txt']);
    Y1 = readmatrix([pathstr '\data\PSG\MASS_SS1_O2CLE_scattering.txt']);
    t1 = readmatrix([pathstr '\data\Stage\MASS_SS1_stage.txt']);
    epoch_num1 = readmatrix([pathstr '\data\Stage\MASS_SS1_epoch_number.txt']);

end
%% Choose the target dataset


if (target_data == 1)
    %% Import SC
    
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])
    Nsubt = 39; % numer of subjects
    
    X2 = readmatrix([pathstr '\data\PSG\SC_FPzCz_scattering.txt']);
    Y2 = readmatrix([pathstr '\data\PSG\SC_PzOz_scattering.txt']);
    t2 = readmatrix([pathstr '\data\Stage\SC_stage.txt']);
    epoch_num2 = readmatrix([pathstr '\data\Stage\SC_epoch_number.txt']);
elseif (target_data ==2)
    %% Import ST  
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])

    Nsubt = 22; % numer of subjects
    X2 = readmatrix([pathstr '\data\PSG\ST_FPzCz_scattering.txt']);
    Y2 = readmatrix([pathstr '\data\PSG\ST_PzOz_scattering.txt']);
    t2 = readmatrix([pathstr '\data\Stage\ST_stage.txt']);
    epoch_num2 = readmatrix([pathstr '\data\Stage\ST_epoch_number.txt']);
elseif (target_data ==3)
    addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])

    Nsubt = 20; % numer of subjects
    X2 = readmatrix([pathstr '\data\PSG\DREAM_O1A2_scattering.txt']);
    Y2 = readmatrix([pathstr '\data\PSG\DREAM_O2A1_scattering.txt']);
    t2 = readmatrix([pathstr '\data\Stage\DREAM_stage.txt']);
    epoch_num2 = readmatrix([pathstr '\data\Stage\DREAM_epoch_number.txt']);

    elseif (target_data == 4)
addpath([pathstr '/sleepedfx_utils/'])
    addpath([pathstr '/edf_reader/'])

    Nsubt = 47; % numer of subjects
    X2 = readmatrix([pathstr '\data\PSG\MASS_SS1_O1CLE_scattering.txt']);
    Y2 = readmatrix([pathstr '\data\PSG\MASS_SS1_O2CLE_scattering.txt']);
    t2 = readmatrix([pathstr '\data\Stage\MASS_SS1_stage.txt']);
    epoch_num2 = readmatrix([pathstr '\data\Stage\MASS_SS1_epoch_number.txt']);
end

%%
epoch_num = [epoch_num1 ; epoch_num2];
sub = Nsubs+Nsubt;
Info.PID = [1:sub]'; 
subjectId = cell(size(Info.PID));
for i = 1:sub
    subjectId{i} = repelem(Info.PID(i), epoch_num(i))';
end

%% merge two hospital data  
AllX = [X1;X2]; 
AllY = [Y1;Y2]; 
Allt = [t1;t2];

%% Construct cvp
[~, response] = max(Allt, [], 2);
% create leave-one-out cross validation partition

cvp = struct;
cvp.NumObservations = size(response, 1);
cvp.testSize = zeros(1, sub);
cvp.trainSize = zeros(1, sub);
cvp.testStore = cell(1, sub);
cvp.trainStore = cell(1, sub);
for i = 1:sub
    cvp.testStore{i} = cell2mat(subjectId) == i;
    cvp.testSize(i) = sum(cvp.testStore{i});
    cvp.trainSize(i) = cvp.NumObservations - cvp.testSize(i);
    cvp.trainStore{i} = ~cvp.testStore{i};
end


nSource = sum(cvp.testSize(1:Nsubs));
nTarget = cvp.NumObservations - nSource;

%% Transfer learning
%cd('C:\Users\l1610\OneDrive\桌面\ming_hsiou')
%col1 = [1;2;3;4;5;6;7;8;9;10];
%outputfile = '10run_SC(300)_to_ST_linear.xlsx';
%xlswrite(outputfile,col1,1,'A2');
%row1 = {'sub1','sub2','sub3','sub4','sub5','sub6','sub7','sub8','sub9','sub10','sub11',...
%    'Average acc','Median acc','Kappa'};
    
%xlswrite(outputfile,row1,1,'B1');
%index_acc_cv = {'B2','B3','B4','B5','B6','B7','B8','B9','B10','B11'};
%index_acc = {'M2','M3','M4','M5','M6','M7','M8','M9','M10','M11'};
%index_Acc_cv = {'N2','N3','N4','N5','N6','N7','N8','N9','N10','N11'};
%index_Kappa = {'O2','O3','O4','O5','O6','O7','O8','O9','O10','O11'};

    
    %each_run_prediction = cell(10,1);
%for i = 1:10
      disp('peko')
if (transfer_learning == 1) %Subspace alignment
   dim_reduction_method = 'SA';
    X1_norm = normalize(X1);
    X2_norm = normalize(X2);
    Y1_norm = normalize(Y1);
    Y2_norm = normalize(Y2);
    [X1_align , X2_align] = subspace_align(X1_norm,X2_norm,50);
    [Y1_align , Y2_align] = subspace_align(Y1_norm,Y2_norm,50);
    AllX_align = [X1_align;X2_align];
    AllY_align = [Y1_align;Y2_align]; 
    predictors = [AllX_align AllY_align];
    
elseif(transfer_learning == 2)
    dim_reduction_method = 'SSTCA';
[Dis1x] = squareform(pdist(AllX,'euclidean')); 
    Sx = reshape(Dis1x,cvp.NumObservations^2,1);
    Sx = sort(Sx,'descend');
    Sx = Sx(1:(cvp.NumObservations^2-cvp.NumObservations)/2);
    sigmaX = median(Sx);
  clear Dis1x Sx
    [Dis1y] = squareform(pdist(AllY,'euclidean'));
    Sy = reshape(Dis1y,cvp.NumObservations^2,1);
    Sy = sort(Sy,'descend');
    Sy = Sy(1:(cvp.NumObservations^2-cvp.NumObservations)/2);
    sigmaY = median(Sy);

clear  Dis1y  Sy
    response_source = response(1:nSource);
    [X1_sstca , X2_sstca] = mySSTCA_reduceKer(X1,X2,response_source,...
        'linear','linear',1,sigmaX,0,1,50,1);
    [Y1_sstca , Y2_sstca] = mySSTCA_reduceKer(Y1,Y2,response_source,...
        'linear','linear',1,sigmaY,0,1,50,1);
    AllX_sstca = [X1_sstca;X2_sstca];
    AllY_sstca = [Y1_sstca;Y2_sstca]; 
    predictors = [AllX_sstca AllY_sstca];
    
elseif(transfer_learning == 3)
   dim_reduction_method = 'rTCA';
    [Dis1x] = squareform(pdist(AllX,'euclidean')); 
    Sx = reshape(Dis1x,cvp.NumObservations^2,1);
    Sx = sort(Sx,'descend');
    Sx = Sx(1:(cvp.NumObservations^2-cvp.NumObservations)/2);
    sigmaX = median(Sx);
  clear Dis1x Sx
    [Dis1y] = squareform(pdist(AllY,'euclidean'));
    Sy = reshape(Dis1y,cvp.NumObservations^2,1);
    Sy = sort(Sy,'descend');
    Sy = Sy(1:(cvp.NumObservations^2-cvp.NumObservations)/2);
    sigmaY = median(Sy);

clear  Dis1y  Sy
    %[X1_tca , X2_tca] = myTCA_reduceKer(X1,X2,'linear',1,1,50,1);
    %[Y1_tca , Y2_tca] = myTCA_reduceKer(Y1,Y2,'linear',1,1,50,1);
    [X1_rtca , X2_rtca] = myrTCA(X1,X2,'linear',0.001,1,1,1);
    [Y1_rtca , Y2_rtca] = myrTCA(Y1,Y2,'linear',0.001,1,1,1);
    AllX_rtca = [X1_rtca;X2_rtca];
    AllY_rtca = [Y1_rtca;Y2_rtca]; 
    predictors = [AllX_rtca AllY_rtca];

elseif(transfer_learning == 0)
    dim_reduction_method = 'CCA';
    display('running CCA . . .') 
    AllX = AllX - repmat(mean(AllX),size(AllX,1),1);
    AllY = AllY - repmat(mean(AllY),size(AllY,1),1);
    [U, ~, V] = svds((AllX')*AllY, 1);
    XU = AllX * real(U);
    YV = AllY * real(V);
    predictors = [XU, YV];
    clear XU YV U V
    display('End of CCA')
    
%
end
%% 5 class SVM
template = templateSVM('KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], 'KernelScale', 10, ...
    'BoxConstraint', 1, 'Standardize', true);

%SVM
response_target = response(nSource+1 : cvp.NumObservations);
Mdl = fitcecoc(predictors(1 : nSource, :), ...
    response(1 : nSource, :), ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', [1; 2; 3; 4; 5]);
%[~, Score] = predict(Mdl, predictors(nSource+1 : cvp.NumObservations, :));
%[~, prediction] = max(Score, [], 2);
%acc = sum(response_target == prediction)/size(response_target,1);
%disp(['Accuracy = ' num2str(acc)])

%each_run_prediction{i,1} = prediction;


% predict on each night
prediction_cv = cell(Nsubt,1);
for j = 1: Nsubt
    response_target = response(cvp.testStore{j+Nsubs});
    [~, validationScores] = predict(Mdl, predictors(cvp.testStore{j+Nsubs}, :));
    [~, prediction_cv{j}] = max(validationScores, [], 2);
    acc_cv(j) = sum(response_target == prediction_cv{j})/size(response_target,1);
end

Acc_median = median(sort(acc_cv,'ascend'));
Acc_average = mean(acc_cv);
writematrix(acc_cv,result_file);
% confusion matrix
cm = cell(Nsubt, 1);
for j = 1:Nsubt
     YY = Allt(cvp.testStore{Nsubs+j},:);
     VV = prediction_cv{j}; 
     VV = full(ind2vec(double(VV'),5))';
    %[~, cm{i}, ~, ~] = confusion(YY', VV');    
    [~, cm{j}, ~, ] = confusion(YY', VV');
end

ConfMat = zeros(5);
for j = 1:size(cm,1)
    ConfMat = ConfMat + cm{j};
end   

%% metrics

SUM=sum(ConfMat,2); %????????
nonzero_idx=find(SUM~=0);
normalizer=zeros(5,1);
normalizer(nonzero_idx)=SUM(nonzero_idx).^(-1); %1/????????
matHMM=diag(normalizer)*ConfMat; %??????
normalized_confusion_matrix = matHMM;

SUM=sum(ConfMat,1);
nonzero_idx=find(SUM~=0);
normalizer=zeros(5,1);
normalizer(nonzero_idx)=SUM(nonzero_idx).^(-1);
normalized_sensitivity_matrix=ConfMat*diag(normalizer);

recall = diag(normalized_confusion_matrix); %hw done
%recall = TP/All real positive
precision = diag(normalized_sensitivity_matrix); %hw done
%precision = TP/All predicted positive

F1_score = 2*(recall.*precision)./(recall+precision); %hw done
Macro_F1 = mean(F1_score);

TOTAL_EPOCH = sum(sum(ConfMat));
ACC = sum(diag(ConfMat))/TOTAL_EPOCH;
EA = sum(sum(ConfMat,1).*sum(transpose(ConfMat),1))/TOTAL_EPOCH^2;%?
kappa = (ACC-EA)/(1-EA); %Cohen’s Kappa
disp(['Cohen’s Kappa = ' num2str(kappa)])

%plot
output = cell(8, 9);
output(1, 2:end) = {'Predict-W', 'Predict-REM', 'Predict-N1', 'Predict-N2', 'Predict-N3', 'PR', 'RE', 'F1'};
output(2:6, 1) = {'Target-W', 'Target-REM', 'Target-N1', 'Target-N2', 'Target-N3'};
output(2:6, 2:6) = num2cell(ConfMat);
output(2:6, 7) = num2cell(precision);
output(2:6, 8) = num2cell(recall);
output(2:6, 9) = num2cell(F1_score);
output(8, 1:3) = {['Accuracy: ' num2str(ACC)], ['Macro F1: ' num2str(Macro_F1)], ['Kappa: ' num2str(kappa)]};
time = clock;
%xlswrite(['C:\Users\John\OneDrive\Code\Sleep\Metrics' num2str(time(4)) num2str(time(5)) '.xls'], output);
 ratio = sum(ConfMat,2)/sum(sum(ConfMat));
    matHMM=normalized_confusion_matrix;

figure;
imagesc(matHMM);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(matHMM(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:5);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(matHMM(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors


%ratio=[numel(find(testing_GT==1)) numel(find(testing_GT==2)) numel(find(testing_GT==3)) numel(find(testing_GT==4)) numel(find(testing_GT==5))]'/length(testing_GT);

ratio=cellstr(num2str(ratio*100,'%5.0f%%'));

delta=0.2;
specialaxis=[1 1+delta 2 2+delta 3 3+delta 4 4+delta 5 5+delta];
set(gca,'XTick',1:5,...                         %# Change the axes tick marks
        'XTickLabel',{'Awake','REM','N1','N2','N3'},...  %#   and tick labels
        'YTick',specialaxis,...
        'YTickLabel',{'Awake',ratio{1},'REM',ratio{2},'N1',ratio{3},'N2',ratio{4},'N3',ratio{5}},...
        'TickLength',[0 0]);
set(gca,'XAxisLocation','top');
%ylabel('Ground Truth');
xlabel(plot_name, 'fontweight','bold');   
% excel
%xlswrite(outputfile,acc_cv,1,char(index_acc_cv(i)));
%xlswrite(outputfile,acc,1,char(index_acc(i)));
%xlswrite(outputfile,Acc_cv,1,char(index_Acc_cv(i)));
%xlswrite(outputfile,kappa,1,char(index_Kappa(i)));
%end
toc

% voting
%prediction_m = zeros(10,size(X2,1));
%for i = 1:size(X2,1)
%    prediction_m(i,:) =  each_run_prediction{i,:};
%end

%for i = 1:size(X2,1)
%    voting_p(i) =  mode(prediction_m(:,i));
%end

%voting_p = voting_p';
%vot = [zeros(size(X1,1),1);voting_p];

%for j = 1: size(t2,1)
%    response_target_v = response(cvp.testStore{j+size(t1,1)});
%    prediction_mode = vot(cvp.testStore{j+size(t1,1)});
%    acc_vote(j) = sum(response_target_v == prediction_mode)/size(response_target_v,1);
%end

%Acc_vote = median(sort(acc_vote,'ascend'));
%acc_vote_average = sum(response_target == voting_p)/size(response_target,1)

%plot
%y = 1:850;
%y = y';
%plot(y,(prediction_m(1,1:850)'),'-*',...
%    y,(prediction_m(2,1:850)'),'-*',...
%    y,(prediction_m(3,1:850)'),'-*');
%fid = fopen('run1_prediction.txt', 'w');
%%fprintf(fid,'%2.0f\n',each_run_prediction{1,1}(1:size(t2{1},1)));
%fclose(fid);

%fid = fopen('run1_prediction.txt', 'r');
%myData = fscanf(fid, '%g');
%fclose(fid);

%% 2D plot
if (2D_plot == 1)
source_sub = predictors(cvp.testStore{2},:);
source_stage = response(cvp.testStore{2});
s_awake_idx = (source_stage == 1);
s_rem_idx = (source_stage == 2);
s_n1_idx = (source_stage == 3); 
s_n2_idx = (source_stage == 4);
s_n3_idx = (source_stage == 5);

target_sub = predictors(cvp.testStore{1+Nsubs},:);
target_stage = response(cvp.testStore{1+Nsubs});

t_awake_idx = (target_stage == 1);
t_rem_idx = (target_stage == 2);
t_n1_idx = (target_stage == 3);
t_n2_idx = (target_stage == 4);
t_n3_idx = (target_stage == 5);


%plot3((source_sub(:,1)),(source_sub(:,2)),(source_sub(:,3)),'rsquare')
%hold on
%plot3((target_sub(:,1)),(target_sub(:,2)),(target_sub(:,3)),'bsquare')

%plot3((source_sub(s_awake_idx,1)),(source_sub(s_awake_idx,2)),(source_sub(s_awake_idx,3)),'rsquare')
%hold on
%plot3((target_sub(t_awake_idx,1)),(target_sub(t_awake_idx,2)),(target_sub(t_awake_idx,3)),'bsquare')
%hold on

%plot3((source_sub(s_rem_idx,1)),(source_sub(s_rem_idx,2)),(source_sub(s_rem_idx,3)),'rsquare')
%hold on
%plot3((target_sub(t_rem_idx,1)),(target_sub(t_rem_idx,2)),(target_sub(t_rem_idx,3)),'bsquare')
%hold on

%plot3((source_sub(s_n1_idx,1)),(source_sub(s_n1_idx,2)),(source_sub(s_n1_idx,3)),'rsquare')
%hold on
%plot3((target_sub(t_n1_idx,1)),(target_sub(t_n1_idx,2)),(target_sub(t_n1_idx,3)),'bsquare')
%hold on

%plot3((source_sub(s_n2_idx,1)),(source_sub(s_n2_idx,2)),(source_sub(s_n2_idx,3)),'rsquare')
%hold on
%plot3((target_sub(t_n2_idx,1)),(target_sub(t_n2_idx,2)),(target_sub(t_n2_idx,3)),'bsquare')
%hold on

%plot3((source_sub(s_n3_idx,1)),(source_sub(s_n3_idx,2)),(source_sub(s_n3_idx,3)),'rsquare')
%hold on
%plot3((target_sub(t_n3_idx,1)),(target_sub(t_n3_idx,2)),(target_sub(t_n3_idx,3)),'bsquare')
%hold on



plot((source_sub(:,1)),(source_sub(:,2)),'rsquare')
hold on
plot((target_sub(:,1)),(target_sub(:,2)),'bsquare')
legend({'source data','target data'},'FontSize',12,'Interpreter','latex','Location','best')

subplot(2,5,1)
plot((source_sub(s_rem_idx,1)),(source_sub(s_rem_idx,2)),'rsquare')
hold on
plot((target_sub(t_rem_idx,1)),(target_sub(t_rem_idx,2)),'bsquare')
legend({'source data','target data'},'FontSize',12,'Interpreter','latex','Location','best')
title(['Awake, ' dim_reduction_method])

subplot(2,5,2)
plot((source_sub(s_n1_idx,1)),(source_sub(s_n1_idx,2)),'rsquare')
hold on
plot((target_sub(t_n1_idx,1)),(target_sub(t_n1_idx,2)),'bsquare')
hold on
legend({'source data','target data'},'FontSize',12,'Interpreter','latex','Location','best')
title(['REM, ' dim_reduction_method])

subplot(2,5,3)
plot((source_sub(s_n2_idx,1)),(source_sub(s_n2_idx,2)),'rsquare')
hold on
plot((target_sub(t_n2_idx,1)),(target_sub(t_n2_idx,2)),'bsquare')
hold on
legend({'source data','target data'},'FontSize',12,'Interpreter','latex','Location','best')
title(['N1, ' dim_reduction_method])

subplot(2,5,4)
plot((source_sub(s_n3_idx,1)),(source_sub(s_n3_idx,2)),'rsquare')
hold on
plot((target_sub(t_n3_idx,1)),(target_sub(t_n3_idx,2)),'bsquare')
hold on
legend({'source data','target data'},'FontSize',12,'Interpreter','latex','Location','best')
title(['N2, ' dim_reduction_method])

subplot(2,5,5)
plot((source_sub(s_awake_idx,1)),(source_sub(s_awake_idx,2)),'rsquare')
hold on
plot((target_sub(t_awake_idx,1)),(target_sub(t_awake_idx,2)),'bsquare')
hold on
legend({'source data','target data'},'FontSize',12,'Interpreter','latex','Location','best')
title(['N3, ' dim_reduction_method])

sgtitle('Data distribution in 2D')
end
