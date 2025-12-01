% -----------------------------------------------------------------
% PHASE 3: EVALUATE MODEL (Full Version with ALL Fixes)
% -----------------------------------------------------------------
clearvars -except X_train Y_train X_test Y_test windowSize binaryMap classOrder trained_model;
clc;
fprintf('Phase 3: Evaluating model on test set...\n');

% --- Prerequisite Check ---
if ~exist('trained_model', 'var')
    error('`trained_model` not found. Please complete Phase 2 and export the model.');
end
if ~exist('X_test', 'var') || ~exist('Y_test', 'var')
    error('`X_test` or `Y_test` not found. Please run the create_split_data.m script first.');
end
if ~exist('classOrder', 'var')
     classOrder = {'H', 'E', 'C'}; % Define class order if not in workspace
end

% --- Step 1: Get Model Predictions & Scores ---
[Y_pred, scores] = trained_model.predictFcn(X_test);

% --- FIX for scores data type ---
% Checks if 'scores' is a cell array and converts it.
% If it's already numeric, it does nothing.
if iscell(scores)
    fprintf('Converting scores from cell to numeric matrix...\n');
    scores = cell2mat(scores);
end
% --- End of Fix ---

% --- Step 2: Compute Total Confusion Matrix ---
C_total = confusionmat(Y_test, Y_pred, 'Order', classOrder);

fprintf('Testing complete. Total Confusion Matrix:\n');
disp(C_total);

% --- Step 3: Final Per-Class Metrics (Sens, Spec, F1) ---
fprintf('\n--- Final Model Performance ---\n');

% Calculate Overall Q3 Accuracy
total_correct = sum(diag(C_total));
total_all = sum(C_total, 'all');
Overall_Accuracy = total_correct / total_all;
fprintf('Overall Q3 Accuracy: %.4f\n\n', Overall_Accuracy);

for k = 1:length(classOrder)
    className = classOrder{k}; % Use curly braces for cell array
    
    % True Positives, False Positives, False Negatives, True Negatives
    TP = C_total(k, k);
    FN = sum(C_total(k, :)) - TP;
    FP = sum(C_total(:, k)) - TP;
    TN = total_all - (TP + FP + FN);
    
    % Sensitivity (Recall)
    if (TP + FN) == 0, Sensitivity = 0; else, Sensitivity = TP / (TP + FN); end
    
    % Specificity
    if (TN + FP) == 0, Specificity = 0; else, Specificity = TN / (TN + FP); end
    
    % Precision (PPV)
    if (TP + FP) == 0, Precision = 0; else, Precision = TP / (TP + FP); end
    
    % F1-Score (F-Measure)
    if (Precision + Sensitivity) == 0
        F1_score = 0;
    else
        F1_score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity);
    end
    
    % Matthews Correlation Coefficient (MCC)
    mcc_num = (TP * TN) - (FP * FN);
    mcc_den = sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) );
    if mcc_den == 0, MCC = 0; else, MCC = mcc_num / mcc_den; end

    % Print all metrics for this class
    fprintf('--- Class: %c ---\n', className);
    fprintf('  Sensitivity (Recall): %.4f\n', Sensitivity);
    fprintf('  Specificity:          %.4f\n', Specificity);
    fprintf('  Precision (PPV):      %.4f\n', Precision);
    fprintf('  F1-Score:             %.4f\n', F1_score);
    fprintf('  MCC:                  %.4f\n', MCC);
end

% --- Step 4: AUC Calculation (One-vs-All) ---
fprintf('\n--- AUC (One-vs-All) ---\n');

% --- FIX for ClassNames property ---
score_order = cellstr(trained_model.ClassificationNeuralNetwork.ClassNames);% --- End of Fix ---

for k = 1:length(classOrder)
    positive_class = classOrder{k};
    
    % Find which column in 'scores' corresponds to our positive class
    score_column_index = find(strcmp(score_order, positive_class));
    
    % Calculate AUC
    [~, ~, ~, AUC_k] = perfcurve(Y_test, scores(:, score_column_index), positive_class);    fprintf('AUC for Class %c (vs. All): %.4f\n', positive_class, AUC_k);
end

fprintf('Phase 3 Complete.\n');

% Clean up intermediate variables
clearvars -except X_train Y_train X_test Y_test trained_model C_total classOrder;