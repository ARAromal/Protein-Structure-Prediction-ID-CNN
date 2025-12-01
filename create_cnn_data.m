% -----------------------------------------------------------------
% CREATE 80/20 CNN-READY DATA (Window Size 5)
% -----------------------------------------------------------------
% This script loads pr_data.m and creates:
% XTrain_cell, YTrain_cat, XTest_cell, YTest_cat
% Data format: Cell array, each cell is [TimeSteps x Channels] = [5 x 5]
% -----------------------------------------------------------------
clear; clc;
fprintf('Starting CNN data preparation (WindowSize=5)...\n');

% --- Step 0: Define Hyperparameters ---
TRAIN_SPLIT_RATIO = 0.8;
windowSize = 5; % As requested
encodingLength = 5; % 5-bit binary code
halfWindow = floor(windowSize / 2);
padChar = 'X';
classOrder = {'H', 'E', 'C'};
rng(42); % For reproducible split

% --- Step 1: Load Data and Map to 3 Classes ---
pr_data; % Loads 'Seq' and 'Str'

% --- 3-Class Mapping ---
map_H = 'HG';
map_E = 'EB';
Str3Class = cell(size(Str));
for i = 1:length(Str)
    s = Str{i};
    s(ismember(s, map_H)) = 'H';
    s(ismember(s, map_E)) = 'E';
    s(~ismember(s, 'HE')) = 'C'; % Map all else to C
    Str3Class{i} = s;
end

% --- Create Binary Map ---
aa = 'ARNDCQEGHILKMFPSTWYVX'; % 21 chars
codes = dec2bin(0:20, 5) - '0';
binaryMap = containers.Map(num2cell(aa), num2cell(codes, 2)');

% --- Step 2: Split Data into 80/20 Sets ---
numProteins = length(Seq);
shuffledIndices = randperm(numProteins);
splitPoint = floor(TRAIN_SPLIT_RATIO * numProteins);
trainIndices = shuffledIndices(1:splitPoint);
testIndices = shuffledIndices(splitPoint+1:end);

% --- Step 3: Process Training Data (80%) ---
fprintf('Processing %d training proteins...\n', length(trainIndices));
XTrain_cell = {};
YTrain_labels = {};

for i = trainIndices
    sequence = Seq{i};
    dssp = Str3Class{i};
    paddedSeq = [repmat(padChar, 1, halfWindow), sequence, repmat(padChar, 1, halfWindow)];
    
    for j = 1:length(sequence)
        window_str = paddedSeq(j:(j + windowSize - 1));
        
        % Create the [TimeSteps x Channels] = [5 x 5] matrix
        window_matrix = zeros(windowSize, encodingLength);
        for k = 1:windowSize
            window_matrix(k, :) = binaryMap(window_str(k));
        end
        
        XTrain_cell{end+1} = window_matrix;
        YTrain_labels{end+1} = dssp(j);
    end
end
YTrain_cat = categorical(YTrain_labels', classOrder);

% --- Step 4: Process Test Data (20%) ---
fprintf('Processing %d test proteins...\n', length(testIndices));
XTest_cell = {};
YTest_labels = {};

for i = testIndices
    sequence = Seq{i};
    dssp = Str3Class{i};
    paddedSeq = [repmat(padChar, 1, halfWindow), sequence, repmat(padChar, 1, halfWindow)];
    
    for j = 1:length(sequence)
        window_str = paddedSeq(j:(j + windowSize - 1));
        
        % Create the [TimeSteps x Channels] = [5 x 5] matrix
        window_matrix = zeros(windowSize, encodingLength);
        for k = 1:windowSize
            window_matrix(k, :) = binaryMap(window_str(k));
        end
        
        XTest_cell{end+1} = window_matrix;
        YTest_labels{end+1} = dssp(j);
    end
end
YTest_cat = categorical(YTest_labels', classOrder);

fprintf('CNN data preparation complete.\n');
clearvars -except XTrain_cell YTrain_cat XTest_cell YTest_cat classOrder;