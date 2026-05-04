%% sEEG_decode.m
% =========================================================================
% Purpose:  sEEG (stereoelectroencephalography) binary decoding pipeline.
%           Performs LDA classification with leave-one-segment-out (LOSO)
%           cross-validation, cost-sensitive learning (uniform priors),
%           and segment-level permutation testing.
%
% Author:   [Your Name]
% Date:     [Date]
%
% Inputs:   A .mat file in a user-specified folder containing:
%   - inputData   : [nObservations x nFeatures] cell array. Each cell holds
%                   a scalar double (averaged power of one 30-s epoch).
%   - decodeScore : [nObservations x 1] double. Continuous scores (1-10).
%   - session_id  : [nObservations x 1] double. Integer segment IDs.
%
% Outputs:  A 'results' structure saved as a .mat file (see Section 6 of
%           the specification for full field descriptions).
%
% Requirements: MATLAB R2020b or later, Statistics and Machine Learning
%               Toolbox, Parallel Computing Toolbox (for permutations).
% =========================================================================

%% ===== USER CONFIGURATION ==============================================

% Path to the folder containing the .mat data file (one .mat file only)
dataFolder = 'E:\8LFP\7MVA\04_YJJ\allsite\decoding_data_alpha';

% Median split option:
%   1 = score >= median -> 1, score < median -> 0
%   2 = score >  median -> 1, score <= median -> 0
medianSplitOption = 2;

% Number of permutations for significance testing
nPermutations = 1000;

% Number of parallel workers for permutation testing
nCores = 24;

%% ===== LOAD DATA ========================================================
fprintf('=== sEEG Decoding Pipeline ===\n');
fprintf('Loading data from: %s\n', dataFolder);

% Automatically find the .mat file in the specified folder
matFiles = dir(fullfile(dataFolder, '*.mat'));
if isempty(matFiles)
    error('No .mat file found in the specified folder: %s', dataFolder);
end
if numel(matFiles) > 1
    % Filter out any results files we may have saved previously
    isResult = contains({matFiles.name}, 'decoding_results_');
    dataFiles = matFiles(~isResult);
    if numel(dataFiles) ~= 1
        error('Expected exactly one data .mat file in folder (found %d). Folder: %s', ...
            numel(dataFiles), dataFolder);
    end
    matFiles = dataFiles;
end

matFilePath = fullfile(dataFolder, matFiles(1).name);
fprintf('Loading file: %s\n', matFiles(1).name);
loadedData = load(matFilePath);

%% ===== INPUT VALIDATION =================================================
% Check that required variables exist
requiredVars = {'inputData', 'decodeScore', 'session_id'};
for v = 1:numel(requiredVars)
    if ~isfield(loadedData, requiredVars{v})
        error('Required variable "%s" not found in the .mat file.', requiredVars{v});
    end
end

inputData  = loadedData.inputData;
decodeScore = loadedData.decodeScore;
session_id  = loadedData.session_id;

% Convert cell array to numeric matrix
% Each cell contains a single scalar double (averaged epoch power)
if iscell(inputData)
    X = cell2mat(inputData);
else
    X = inputData;  % Already numeric
end

[nObservations, nFeatures] = size(X);

% Validate dimensions
assert(numel(decodeScore) == nObservations, ...
    'decodeScore length (%d) does not match inputData rows (%d).', ...
    numel(decodeScore), nObservations);
assert(numel(session_id) == nObservations, ...
    'session_id length (%d) does not match inputData rows (%d).', ...
    numel(session_id), nObservations);

% Ensure column vectors
decodeScore = decodeScore(:);
session_id  = session_id(:);

% Check for exactly 15 unique segments
uniqueSegments = unique(session_id);
nSegments = numel(uniqueSegments);
assert(nSegments == 10, ...
    'Expected 10 unique segments, found %d.', nSegments);%%%%%%%%%%%%%%%%%%%%%%%%%

% Validate that all observations within each segment share the same score
for s = 1:nSegments
    segMask = (session_id == uniqueSegments(s));
    segScores = unique(decodeScore(segMask));
    assert(numel(segScores) == 1, ...
        'Segment %d has multiple different decodeScore values.', uniqueSegments(s));
end

fprintf('Data loaded: %d observations, %d features, %d segments\n', ...
    nObservations, nFeatures, nSegments);

%% ===== COMPUTE SEGMENT-LEVEL INFORMATION ================================
% Extract one score per segment and compute observation counts
segmentScores = zeros(nSegments, 1);
segmentSizes  = zeros(nSegments, 1);
for s = 1:nSegments
    segMask = (session_id == uniqueSegments(s));
    segmentScores(s) = decodeScore(find(segMask, 1));
    segmentSizes(s)  = sum(segMask);
end

%% ===== MEDIAN SPLIT BINARISATION ========================================
% Compute the median across the 15 unique segment-level scores
medianValue = median(segmentScores);
fprintf('Median of segment scores: %.4f\n', medianValue);

% Assign binary labels at the segment level based on the chosen option
segmentLabels = zeros(nSegments, 1);
if medianSplitOption == 1
    % Option 1: score >= median -> 1, score < median -> 0
    segmentLabels = double(segmentScores >= medianValue);
    fprintf('Median split option 1: score >= %.4f -> 1, score < %.4f -> 0\n', ...
        medianValue, medianValue);
elseif medianSplitOption == 2
    % Option 2: score > median -> 1, score <= median -> 0
    segmentLabels = double(segmentScores > medianValue);
    fprintf('Median split option 2: score > %.4f -> 1, score <= %.4f -> 0\n', ...
        medianValue, medianValue);
else
    error('Invalid medianSplitOption: %d. Must be 1 or 2.', medianSplitOption);
end

% Propagate segment-level binary labels to all observations
binaryLabels = zeros(nObservations, 1);
for s = 1:nSegments
    segMask = (session_id == uniqueSegments(s));
    binaryLabels(segMask) = segmentLabels(s);
end

nClass0_seg = sum(segmentLabels == 0);
nClass1_seg = sum(segmentLabels == 1);
fprintf('Class balance (segments): class 0 = %d, class 1 = %d\n', ...
    nClass0_seg, nClass1_seg);

%% ===== SET UP PARALLEL POOL =============================================
fprintf('Setting up parallel pool with %d workers...\n', nCores);
currentPool = gcp('nocreate');
if ~isempty(currentPool)
    if currentPool.NumWorkers ~= nCores
        fprintf('Existing pool has %d workers; deleting and creating new pool.\n', ...
            currentPool.NumWorkers);
        delete(currentPool);
        parpool('local', nCores);
    else
        fprintf('Existing pool with %d workers is already open.\n', nCores);
    end
else
    parpool('local', nCores);
end

%% ===== LOSO CV (ALL 10 SEGMENTS) ========================================
fprintf('\nRunning observed LOSO CV across %d segments...\n', nSegments);

% Pre-build per-segment index arrays
segIndices = cell(nSegments, 1);
for s = 1:nSegments
    segIndices{s} = find(session_id == uniqueSegments(s));
end

% Storage for per-fold results
foldAccuracies         = zeros(nSegments, 1);
foldBalancedAccuracies = zeros(nSegments, 1);
foldSensitivities      = zeros(nSegments, 1);
foldSpecificities      = zeros(nSegments, 1);
foldAUCs               = zeros(nSegments, 1);
foldCoeffs             = zeros(nSegments, nFeatures);

% Collect per-observation predictions and posteriors (each obs appears in exactly one test fold)
allPredictions = NaN(nObservations, 1);
allPosteriors  = NaN(nObservations, 1);

for k = 1:nSegments
    testSegID = uniqueSegments(k);
    testMask  = (session_id == testSegID);
    trainMask = ~testMask;

    X_train = X(trainMask, :);
    y_train = binaryLabels(trainMask);
    X_test  = X(testMask, :);
    y_test  = binaryLabels(testMask);

    % Cost-sensitive LDA: uniform priors balance each class's influence on the
    % decision boundary regardless of the (imbalanced) training class counts.
    mdl = fitcdiscr(X_train, y_train, 'Prior', 'uniform');

    [yPred, posteriorProbs] = predict(mdl, X_test);

    class1Idx = find(mdl.ClassNames == 1);
    if isempty(class1Idx)
        error('Class 1 not found in model ClassNames for fold %d.', k);
    end
    posteriorClass1 = posteriorProbs(:, class1Idx);

    allPredictions(testMask) = yPred;
    allPosteriors(testMask)  = posteriorClass1;

    % Per-fold metrics
    TP_k = sum(yPred == 1 & y_test == 1);
    TN_k = sum(yPred == 0 & y_test == 0);
    FP_k = sum(yPred == 1 & y_test == 0);
    FN_k = sum(yPred == 0 & y_test == 1);

    sens_k = TP_k / max(TP_k + FN_k, 1);
    spec_k = TN_k / max(TN_k + FP_k, 1);

    foldAccuracies(k)         = (TP_k + TN_k) / numel(y_test);
    foldBalancedAccuracies(k) = (sens_k + spec_k) / 2;
    foldSensitivities(k)      = sens_k;
    foldSpecificities(k)      = spec_k;

    try
        [~, ~, ~, auc_k] = perfcurve(y_test, posteriorClass1, 1);
        foldAUCs(k) = auc_k;
    catch
        foldAUCs(k) = NaN;
    end

    foldCoeffs(k, :) = mdl.Coeffs(1, 2).Linear';

    fprintf('  Fold %d/%d: held-out segment %d, BA=%.4f, AUC=%.4f\n', ...
        k, nSegments, testSegID, foldBalancedAccuracies(k), foldAUCs(k));
end

%% ===== AGGREGATE OBSERVED METRICS ACROSS FOLDS ==========================
fprintf('\nAggregating observed metrics across %d folds...\n', nSegments);

obsAccuracy         = mean(foldAccuracies);
obsBalancedAccuracy = mean(foldBalancedAccuracies);
obsSensitivity      = mean(foldSensitivities);
obsSpecificity      = mean(foldSpecificities);
obsAUC              = nanmean(foldAUCs);

% Aggregate feature weights: mean across all folds
rawWeights = mean(foldCoeffs, 1);  % 1 x nFeatures
maxAbsWeight = max(abs(rawWeights));
if maxAbsWeight == 0
    scaledWeights = zeros(size(rawWeights));
else
    scaledWeights = rawWeights / maxAbsWeight;
end

fprintf('  Accuracy:          %.4f\n', obsAccuracy);
fprintf('  Balanced Accuracy: %.4f\n', obsBalancedAccuracy);
fprintf('  Sensitivity:       %.4f\n', obsSensitivity);
fprintf('  Specificity:       %.4f\n', obsSpecificity);
fprintf('  AUC:               %.4f\n', obsAUC);

%% ===== POOLED POSTERIORS FOR ROC CURVE ==================================
% Each observation is the test set in exactly one LOSO fold — no averaging needed.
pooledPosteriors = allPosteriors;
validObsMask     = ~isnan(pooledPosteriors);

% Save into results
results.observed.posteriors   = pooledPosteriors;
results.observed.trueLabels   = binaryLabels;
results.observed.validObsMask = validObsMask;

fprintf('Pooled posteriors collected for %d/%d observations.\n', ...
    sum(validObsMask), nObservations);

%% ===== PERMUTATION TESTING ==============================================
fprintf('\nRunning %d permutations (segment-level label shuffling, LOSO)...\n', nPermutations);

% Broadcast variables for parfor
X_broadcast = X;
sid_broadcast = session_id;
uSeg_broadcast = uniqueSegments;
nSeg_broadcast = nSegments;
nFeat_broadcast = nFeatures;
segIndices_broadcast = segIndices;
segBinaryLabels = segmentLabels;

% Preallocate output arrays
permBalancedAccuracies = zeros(nPermutations, 1);
permAUCs = zeros(nPermutations, 1);

parfor p = 1:nPermutations
    % --- Segment-level permutation ---
    permOrder = randperm(nSeg_broadcast);
    shuffledSegLabels = segBinaryLabels(permOrder);

    % Reconstruct full permuted label vector
    nObs_p = size(X_broadcast, 1);
    permLabels = zeros(nObs_p, 1);
    for s = 1:nSeg_broadcast
        permLabels(segIndices_broadcast{s}) = shuffledSegLabels(s);
    end

    foldBAs_p  = zeros(nSeg_broadcast, 1);
    foldAUCs_p = zeros(nSeg_broadcast, 1);

    for k = 1:nSeg_broadcast
        testSID = uSeg_broadcast(k);
        tMask   = (sid_broadcast == testSID);
        trMask  = ~tMask;

        y_tr = permLabels(trMask);
        X_tr = X_broadcast(trMask, :);
        X_te = X_broadcast(tMask, :);
        y_te = permLabels(tMask);

        % Skip fold if training set is single-class (can occur under permutation)
        if numel(unique(y_tr)) < 2
            foldBAs_p(k)  = NaN;
            foldAUCs_p(k) = NaN;
            continue;
        end

        % Cost-sensitive LDA: uniform priors (mirrors observed CV)
        mdlP = fitcdiscr(X_tr, y_tr, 'Prior', 'uniform');
        [pred_p, post_p] = predict(mdlP, X_te);

        c1 = find(mdlP.ClassNames == 1);
        if isempty(c1)
            post1 = zeros(size(y_te));
        else
            post1 = post_p(:, c1);
        end

        tp = sum(pred_p == 1 & y_te == 1);
        tn = sum(pred_p == 0 & y_te == 0);
        fp = sum(pred_p == 1 & y_te == 0);
        fn = sum(pred_p == 0 & y_te == 1);
        sens_pp = tp / max(tp + fn, 1);
        spec_pp = tn / max(tn + fp, 1);
        foldBAs_p(k) = (sens_pp + spec_pp) / 2;

        try
            [~, ~, ~, auc_pp] = perfcurve(y_te, post1, 1);
            foldAUCs_p(k) = auc_pp;
        catch
            foldAUCs_p(k) = NaN;
        end
    end

    permBalancedAccuracies(p) = nanmean(foldBAs_p);
    permAUCs(p) = nanmean(foldAUCs_p);

    if mod(p, 100) == 0
        fprintf('  Permutation %d/%d completed.\n', p, nPermutations);
    end
end

%% ===== COMPUTE P-VALUES =================================================
% p = (number of permutation metrics >= observed metric + 1) / (nPermutations + 1)
validPermBA  = permBalancedAccuracies(~isnan(permBalancedAccuracies));
validPermAUC = permAUCs(~isnan(permAUCs));

pValue_balancedAccuracy = (sum(validPermBA >= obsBalancedAccuracy) + 1) / (numel(validPermBA) + 1);
pValue_AUC = (sum(validPermAUC >= obsAUC) + 1) / (numel(validPermAUC) + 1);

fprintf('\nPermutation test results:\n');
fprintf('  Balanced Accuracy p-value: %.4f\n', pValue_balancedAccuracy);
fprintf('  AUC p-value:               %.4f\n', pValue_AUC);

%% ===== BUILD RESULTS STRUCTURE ==========================================
fprintf('\nBuilding results structure...\n');

% Global observed results (aggregated across folds)
results.observed.accuracy          = obsAccuracy;
results.observed.balancedAccuracy  = obsBalancedAccuracy;
results.observed.sensitivity       = obsSensitivity;
results.observed.specificity       = obsSpecificity;
results.observed.AUC               = obsAUC;
results.observed.featureWeights.raw    = rawWeights;
results.observed.featureWeights.scaled = scaledWeights;

% Per-fold results keyed by held-out session_id
for k = nSegments:-1:1
    results.folds(k).heldOutSegment  = uniqueSegments(k);
    results.folds(k).accuracy        = foldAccuracies(k);
    results.folds(k).balancedAccuracy = foldBalancedAccuracies(k);
    results.folds(k).sensitivity     = foldSensitivities(k);
    results.folds(k).specificity     = foldSpecificities(k);
    results.folds(k).AUC             = foldAUCs(k);
    results.folds(k).foldCoefficients = foldCoeffs(k, :);
end

% Permutation results
results.permutation.balancedAccuracies      = permBalancedAccuracies;
results.permutation.AUCs                    = permAUCs;
results.permutation.pValue_balancedAccuracy = pValue_balancedAccuracy;
results.permutation.pValue_AUC             = pValue_AUC;
results.permutation.nPermutations          = nPermutations;

% Metadata
results.info.medianSplitOption = medianSplitOption;
results.info.medianValue       = medianValue;
results.info.nSegments         = nSegments;
results.info.nObservations     = nObservations;
results.info.nFeatures         = nFeatures;
results.info.segmentSizes      = segmentSizes;
results.info.segmentScores     = segmentScores;
results.info.segmentLabels     = segmentLabels;
results.info.classBalance.nClass0 = nClass0_seg;
results.info.classBalance.nClass1 = nClass1_seg;
results.info.dateRun           = datestr(now, 'yyyy-mm-dd HH:MM:SS');

%% ===== SAVE RESULTS =====================================================
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
resultsFilename = sprintf('decoding_results_%s.mat', timestamp);
resultsPath = fullfile(dataFolder, resultsFilename);
save(resultsPath, 'results');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', resultsPath);
fprintf('  Folds:               %d\n', nSegments);
fprintf('  Accuracy:            %.4f\n', results.observed.accuracy);
fprintf('  Balanced Accuracy:   %.4f (p = %.4f)\n', ...
    results.observed.balancedAccuracy, pValue_balancedAccuracy);
fprintf('  AUC:                 %.4f (p = %.4f)\n', ...
    results.observed.AUC, pValue_AUC);
fprintf('  Sensitivity:         %.4f\n', results.observed.sensitivity);
fprintf('  Specificity:         %.4f\n', results.observed.specificity);
fprintf('=============================\n');
