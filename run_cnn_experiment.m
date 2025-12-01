% -----------------------------------------------------------------
% MASTER SCRIPT for 1D CNN Protein Structure Prediction
% -----------------------------------------------------------------
clear; clc;

% --- 1. Prepare Data ---
% This script creates XTrain_cell, YTrain_cat, XTest_cell, YTest_cat
create_cnn_data; 
% We will use the 'Test' set as the 'Validation' set for the function
XTrain = XTrain_cell;
YTrain = YTrain_cat;
XValidation = XTest_cell;
YValidation = YTest_cat;

% --- 2. Define CNN Parameters (Fine-tune here) ---
% These are the "parameters given over there" from your script
cnnParams.InitialLearnRate = 0.001;
cnnParams.MaxEpochs = 30;
cnnParams.MiniBatchSize = 16;
cnnParams.FilterSize = 3; % Kernel size
cnnParams.NumFilters1 = 32; % Filters in first conv layer
cnnParams.NumFilters2 = 64; % Filters in second conv layer

% --- 3. Train and Evaluate the Model ---
[trainedNet, validationMetrics] = trainProteinCNN(XTrain, YTrain, XValidation, YValidation, cnnParams);

% --- 4. Save Results ---
save('cnn_model.mat', 'trainedNet', 'validationMetrics', 'cnnParams');
fprintf('\nExperiment complete. Trained network and metrics saved to cnn_model.mat\n');