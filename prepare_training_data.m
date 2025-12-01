% -----------------------------------------------------------------
% CREATE 80/20 TRAIN/TEST SPLIT
% This script loads pr_data.m and creates X_train, Y_train,
% X_test, and Y_test.
% -----------------------------------------------------------------
clear; clc;
fprintf('Starting data preparation with 80/20 split...\n');

% --- Step 0: Define Hyperparameters ---
TRAIN_SPLIT_RATIO = 0.8;
windowSize = 5;
halfWindow = floor(windowSize / 2);
padChar = 'X';
classOrder = {'H', 'E', 'C'}; % Use cell array for categorical
rng(42); % Use a fixed random seed for reproducible results

% --- Step 1: Load Data and Map to 3 Classes ---
pr_data; % Loads 'Seq' and 'Str'

% --- Data Integrity Check ---
for k = 1:length(Seq)
    if length(Seq{k}) ~= length(Str{k})
        fprintf('\n!!! DATA ERROR: Mismatch in data pair %d !!!\n', k);
        error('Sequence (Seq) and Structure (Str) strings must have the exact same length. Please fix this entry in your pr_data.m file.');
    end
end
fprintf('Data integrity check passed.\n');

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
aa = 'ARNDCQEGHILKMFPSTWYVX'; % 21 chars (20 AAs + 'X' pad)
codes = dec2bin(0:20, 5) - '0';
binaryMap = containers.Map(num2cell(aa), num2cell(codes, 2)');
featureVectorLength = windowSize * 5; % 17 * 5 = 85

% --- Step 2: Split Data into 80/20 Sets ---
numProteins = length(Seq);
shuffledIndices = randperm(numProteins);

splitPoint = floor(TRAIN_SPLIT_RATIO * numProteins);
trainIndices = shuffledIndices(1:splitPoint);
testIndices = shuffledIndices(splitPoint+1:end);

% --- Step 3: Process Training Data (80%) ---
fprintf('Processing %d training proteins...\n', length(trainIndices));
all_train_windows = {};
all_train_labels = {};

for i = trainIndices
    sequence = Seq{i};
    dssp = Str3Class{i};
    paddedSeq = [repmat(padChar, 1, halfWindow), sequence, repmat(padChar, 1, halfWindow)];
    
    for j = 1:length(sequence)
        window_start = j;
        window_end = j + windowSize - 1;
        all_train_windows{end+1} = paddedSeq(window_start:window_end);
        all_train_labels{end+1} = dssp(j);
    end
end

% --- Step 4: Process Test Data (20%) ---
fprintf('Processing %d test proteins...\n', length(testIndices));
all_test_windows = {};
all_test_labels = {};

for i = testIndices
    sequence = Seq{i};
    dssp = Str3Class{i};
    paddedSeq = [repmat(padChar, 1, halfWindow), sequence, repmat(padChar, 1, halfWindow)];
    
    for j = 1:length(sequence)
        window_start = j;
        window_end = j + windowSize - 1;
        all_test_windows{end+1} = paddedSeq(window_start:window_end);
        all_test_labels{end+1} = dssp(j);
    end
end

% --- Step 5: Vectorize Training Data ---
fprintf('Vectorizing training data...\n');
numTrainWindows = length(all_train_windows);
X_train = zeros(numTrainWindows, featureVectorLength);
for i = 1:numTrainWindows
    window_str = all_train_windows{i};
    feature_vector = zeros(1, featureVectorLength);
    for j = 1:windowSize
        binary_code = binaryMap(window_str(j));
        start_idx = (j-1)*5 + 1;
        end_idx = j*5;
        feature_vector(start_idx:end_idx) = binary_code;
    end
    X_train(i, :) = feature_vector;
end
Y_train = categorical(all_train_labels'); % Convert cell to categorical

% --- Step 6: Vectorize Test Data ---
fprintf('Vectorizing test data...\n');
numTestWindows = length(all_test_windows);
X_test = zeros(numTestWindows, featureVectorLength);
for i = 1:numTestWindows
    window_str = all_test_windows{i};
    feature_vector = zeros(1, featureVectorLength);
    for j = 1:windowSize
        binary_code = binaryMap(window_str(j));
        start_idx = (j-1)*5 + 1;
        end_idx = j*5;
        feature_vector(start_idx:end_idx) = binary_code;
    end
    X_test(i, :) = feature_vector;
end
Y_test = categorical(all_test_labels'); % Convert cell to categorical

% --- Step 7: Final Summary ---
fprintf('\n--- Data Preparation Complete ---\n');
fprintf('  Training Set: %d proteins, %d total windows\n', length(trainIndices), size(X_train, 1));
fprintf('  Test Set:     %d proteins, %d total windows\n', length(testIndices), size(X_test, 1));
fprintf('Ready for model training.\n');

% Clean up intermediate variables
clearvars -except X_train Y_train X_test Y_test windowSize binaryMap classOrder;