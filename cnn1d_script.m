% MATLAB Commands for data preparation
% ------------------------------------
% Assume your original data is stored in a MATLAB array called 'Data'
% of size 200 x 18.
% 1. Separate predictors (X) and labels (Y)
%X = Data(:, 1:17); % Predictors: 200 samples, 17 features/time steps
%Y = Data(:, 18);   % Labels: 200 samples
% 2. Convert predictors to a cell array format
% Each cell should contain one sample (17 elements), typically as a column vector or a row vector depending on layer input size configuration.
% For sequenceInputLayer, it expects numTimeSteps-by-numChannels. Here numTimeSteps=17, numChannels=1.
X_sequences = cell(12003, 1);
for i = 1:12003
    X_sequences{i} = X_train(i, :)'; % Transpose to a column vector (17x1)
end
% 3. Convert labels to a categorical array
%Y_categorical = categorical(Y);
Y_categorical = Y_train;
% 4. Split data into training and validation sets (optional but recommended)
cv = cvpartition(Y_categorical, 'HoldOut', 0.2); % 80% train, 20% validation
XTrain = X_sequences(cv.training);
YTrain = Y_categorical(cv.training);
XValidation = X_sequences(cv.test);
YValidation = Y_categorical(cv.test);
% Step 2: Define the 1-D CNN Architecture
% Define the layers of your 1D CNN. The architecture typically involves 1D convolutional layers, ReLU activation, pooling layers, and a final fully connected layer with a softmax and classification layer. 
% 
% MATLAB Commands for 1-D CNN architecture
% ----------------------------------------
% Determine the number of classes
numClasses = numel(categories(Y_categorical));
% Define the network layers
layers = [
    sequenceInputLayer(25) % Input size is the length of one sample (17 features)
    
    convolution1dLayer(3, 3, 'Padding', 'causal') % 32 filters of size 3, 'causal' padding helps with sequence data
    batchNormalizationLayer
    reluLayer
    
    maxPooling1dLayer(1, 'Stride', 2) % Max pooling layer
    
    convolution1dLayer(3, 10, 'Padding', 'causal') % 64 filters of size 3
    batchNormalizationLayer
    reluLayer
    
    globalAveragePooling1dLayer() % Use global pooling to reduce spatial dimensions to a single vector
    
    fullyConnectedLayer(numClasses) % Fully connected layer with output size = number of classes
    softmaxLayer
    classificationLayer
];
% Step 3: Specify Training Options
% Configure the training parameters using the trainingOptions function
% 
% MATLAB Commands for generating options for training
% ---------------------------------------------------
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
% Step 4: Train the Network
% Use the trainNetwork function to train the defined network with the prepared data and specified options
% 
% MATLAB Commands to train the network
% ------------------------------
net = trainNetwork(XTrain, YTrain, layers, options);
% Step 5: Evaluate the Network (Optional)
% After training, you can evaluate the network's performance on the validation or a separate test set. 
% 
% MATLAB Commands to evaluate the network
% ------------------------------
YPred = classify(net, XValidation);
accuracy = sum(YPred == YValidation) / numel(YValidation);
disp(['Validation accuracy: ', num2str(accuracy)]);