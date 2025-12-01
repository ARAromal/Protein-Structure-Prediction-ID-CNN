function [net, metrics] = trainProteinCNN(XTrain, YTrain, XValidation, YValidation, cnnParams)
% trainProteinCNN: Trains a 1D CNN for protein structure prediction.
%
% INPUTS:
%   XTrain:      Cell array of training data (e.g., 5x5 matrices)
%   YTrain:      Categorical array of training labels
%   XValidation: Cell array of validation data
%   YValidation: Categorical array of validation labels
%   cnnParams:   Struct with network and training parameters:
%                .InitialLearnRate
%                .MaxEpochs
%                .MiniBatchSize
%                .NumFilters1
%                .NumFilters2
%                .FilterSize
%
% OUTPUTS:
%   net:         The trained deep learning network
%   metrics:     A table of performance metrics (Accuracy, Sens, Spec, etc.)

fprintf('Defining 1D CNN architecture...\n');

% --- 1. Define Network Architecture ---
numClasses = numel(categories(YTrain));
inputSize = size(XTrain{1}, 2); % Should be 5

layers = [
    % --- THIS IS THE FIX ---
    % We must specify 'MinLength' so the network analyzer 
    % knows the input is long enough for the pooling layer.
    sequenceInputLayer(inputSize, 'Name', 'input', 'MinLength', 5) % Input channels=5, MinLength=5
    % --- END OF FIX ---
    
    convolution1dLayer(cnnParams.FilterSize, cnnParams.NumFilters1, 'Padding', 'causal', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution1dLayer(cnnParams.FilterSize, cnnParams.NumFilters2, 'Padding', 'causal', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    globalAveragePooling1dLayer('Name', 'gapool')
    
    flattenLayer('Name', 'flatten')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% --- 2. Specify Training Options ---
options = trainingOptions('adam', ...
    'InitialLearnRate', cnnParams.InitialLearnRate, ...
    'MaxEpochs', cnnParams.MaxEpochs, ...
    'MiniBatchSize', cnnParams.MiniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 100, ... % Check validation every 100 iterations
    'Verbose', false, ...
    'Plots', 'training-progress');

% --- 3. Train the Network ---
fprintf('Training 1D CNN...\n');
net = trainNetwork(XTrain, YTrain, layers, options);
fprintf('Training complete.\n');

% --- 4. Evaluate and Compute Metrics (Task 2) ---
fprintf('Calculating validation metrics...\n');
YPred = classify(net, XValidation, 'MiniBatchSize', cnnParams.MiniBatchSize);

% Get class order
classOrder = categories(YTrain);
C_total = confusionmat(YValidation, YPred, 'Order', classOrder);

% Initialize metrics storage
numClasses = length(classOrder);
Sensitivity = zeros(numClasses, 1);
Specificity = zeros(numClasses, 1);
Precision = zeros(numClasses, 1);
F1_score = zeros(numClasses, 1);
MCC = zeros(numClasses, 1);

total_all = sum(C_total, 'all');

for k = 1:numClasses
    TP = C_total(k, k);
    FN = sum(C_total(k, :)) - TP;
    FP = sum(C_total(:, k)) - TP;
    TN = total_all - (TP + FP + FN);
    
    % Sensitivity (Recall)
    if (TP + FN) == 0, Sensitivity(k) = 0; else, Sensitivity(k) = TP / (TP + FN); end
    
    % Specificity
    if (TN + FP) == 0, Specificity(k) = 0; else, Specificity(k) = TN / (TN + FP); end
    
    % Precision (PPV)
    if (TP + FP) == 0, Precision(k) = 0; else, Precision(k) = TP / (TP + FP); end
    
    % F1-Score
    if (Precision(k) + Sensitivity(k)) == 0
        F1_score(k) = 0;
    else
        F1_score(k) = 2 * (Precision(k) * Sensitivity(k)) / (Precision(k) + Sensitivity(k));
    end
    
    % Matthews Correlation Coefficient (MCC)
    mcc_num = (TP * TN) - (FP * FN);
    mcc_den = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
    if mcc_den == 0, MCC(k) = 0; else, MCC(k) = mcc_num / mcc_den; end
end

% Overall Accuracy
Overall_Accuracy = sum(diag(C_total)) / total_all;

% --- 5. Format Output Metrics ---
% Create a summary table
metrics = table(Sensitivity, Specificity, Precision, F1_score, MCC, 'RowNames', classOrder);

% Add overall accuracy
OverallMetrics = table(Overall_Accuracy, 'VariableNames', {'Accuracy'}, 'RowNames', {'Overall'});
disp('Validation Confusion Matrix:');
disp(C_total);
disp('Validation Performance Metrics:');
disp(metrics);
fprintf('Overall Accuracy: %.4f\n', Overall_Accuracy);

% Note: We are not returning the OverallMetrics table, but printing it.
% The main 'metrics' table contains the per-class details.
end