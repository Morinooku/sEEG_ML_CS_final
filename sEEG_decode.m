%% sEEG_decode.m
% =========================================================================
% Purpose:  sEEG (stereoelectroencephalography) binary decoding pipeline.
%           Performs LDA classification with combinatorial segment removal,
%           per-fold downsampling, leave-one-segment-out cross-validation,
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

%% ===== ENUMERATE BALANCED COMBINATIONS ==================================
% Identify majority class (9 segments) and minority class (6 segments)
if nClass0_seg > nClass1_seg
    majorityLabel = 0;
else
    majorityLabel = 1;
end

majoritySegIdx = find(segmentLabels == majorityLabel);  % indices into uniqueSegments

% Five fixed groups of 3 consecutive segment IDs
allGroups = {[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]};%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Keep only groups where all 3 segments belong to the majority class
removalCombos = zeros(0, 2);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g = 1:numel(allGroups)
    groupIDs = allGroups{g};
    groupIdx = find(ismember(uniqueSegments, groupIDs));
    if all(ismember(groupIdx, majoritySegIdx))
        removalCombos = [removalCombos; groupIdx(:)'];
    end
end
nCombos = size(removalCombos, 1);
fprintf('Number of balanced removal combinations: %d\n', nCombos);

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

%% ===== OBSERVED LOSO CV ACROSS COMBINATIONS =============================
fprintf('\nRunning observed LOSO CV across %d combinations...\n', nCombos);

% Pre-build per-segment index arrays
segIndices = cell(nSegments, 1);
for s = 1:nSegments
    segIndices{s} = find(session_id == uniqueSegments(s));
end

% Storage for per-combination results
comboAccuracies         = zeros(nCombos, 1);
comboBalancedAccuracies = zeros(nCombos, 1);
comboSensitivities      = zeros(nCombos, 1);
comboSpecificities      = zeros(nCombos, 1);
comboAUCs               = zeros(nCombos, 1);
comboAllCoeffs          = zeros(nCombos, 8, nFeatures);  % nCombos x 12folds x nFeatures%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for c = 1:nCombos
    removeSegIdx = removalCombos(c, :);  % 3 indices into uniqueSegments
    removeSegIDs = uniqueSegments(removeSegIdx);

    % Build mask: keep observations NOT in removed segments
    keepMask = ~ismember(session_id, removeSegIDs);
    X_combo = X(keepMask, :);
    y_combo = binaryLabels(keepMask);
    sid_combo = session_id(keepMask);

    % Remaining 12 segments
    remainingSegIdx = setdiff(1:nSegments, removeSegIdx);
    remainingSegIDs = uniqueSegments(remainingSegIdx);
    nRemainingSegs = numel(remainingSegIDs);

    % Preallocate for this combination
    comboPredictions = NaN(size(y_combo));
    comboPosteriors  = NaN(size(y_combo));
    comboFoldCoeffs  = zeros(nRemainingSegs, nFeatures);

    for k = 1:nRemainingSegs
        testSegID = remainingSegIDs(k);
        testMask_c  = (sid_combo == testSegID);
        trainMask_c = ~testMask_c;

        X_train_raw = X_combo(trainMask_c, :);
        y_train_raw = y_combo(trainMask_c);
        X_test_c    = X_combo(testMask_c, :);

        % Per-fold downsampling: equalize training class observation counts
        idx0 = find(y_train_raw == 0);
        idx1 = find(y_train_raw == 1);
        n0_train = numel(idx0);
        n1_train = numel(idx1);

        if n0_train > n1_train
            % Downsample class 0 to match class 1
            keepIdx0 = idx0(randperm(n0_train, n1_train));
            trainIdx = sort([keepIdx0; idx1]);
        elseif n1_train > n0_train
            % Downsample class 1 to match class 0
            keepIdx1 = idx1(randperm(n1_train, n0_train));
            trainIdx = sort([idx0; keepIdx1]);
        else
            trainIdx = (1:numel(y_train_raw))';
        end

        X_train = X_train_raw(trainIdx, :);
        y_train = y_train_raw(trainIdx);

        % Train plain LDA (no cost matrix, default priors)
        mdl = fitcdiscr(X_train, y_train);

        % Predict on held-out segment
        [yPred, posteriorProbs] = predict(mdl, X_test_c);

        class1Idx = find(mdl.ClassNames == 1);
        if isempty(class1Idx)
            error('Class 1 not found in model ClassNames for combo %d, fold %d.', c, k);
        end
        posteriorClass1 = posteriorProbs(:, class1Idx);

        % Store pooled predictions
        comboPredictions(testMask_c) = yPred;
        comboPosteriors(testMask_c)  = posteriorClass1;

        % Store LDA coefficients
        comboFoldCoeffs(k, :) = mdl.Coeffs(1, 2).Linear';
    end

    % Compute metrics for this combination
    trueLabels_c = y_combo;
    TP_c = sum(comboPredictions == 1 & trueLabels_c == 1);
    TN_c = sum(comboPredictions == 0 & trueLabels_c == 0);
    FP_c = sum(comboPredictions == 1 & trueLabels_c == 0);
    FN_c = sum(comboPredictions == 0 & trueLabels_c == 1);

    sens_c = TP_c / max(TP_c + FN_c, 1);
    spec_c = TN_c / max(TN_c + FP_c, 1);

    comboAccuracies(c)         = (TP_c + TN_c) / numel(trueLabels_c);
    comboBalancedAccuracies(c) = (sens_c + spec_c) / 2;
    comboSensitivities(c)      = sens_c;
    comboSpecificities(c)      = spec_c;

    try
        [~, ~, ~, auc_c] = perfcurve(trueLabels_c, comboPosteriors, 1);
        comboAUCs(c) = auc_c;
    catch
        comboAUCs(c) = NaN;
    end

    comboAllCoeffs(c, :, :) = comboFoldCoeffs;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     fprintf('  Combination %d/%d: removed segments [%d %d %d], BA=%.4f, AUC=%.4f\n', ...
%         c, nCombos, removeSegIDs(1), removeSegIDs(2), removeSegIDs(3), ...
%         comboBalancedAccuracies(c), comboAUCs(c));
    fprintf('  Combination %d/%d: removed segments [%d %d], BA=%.4f, AUC=%.4f\n', ...
        c, nCombos, removeSegIDs(1), removeSegIDs(2), ...
        comboBalancedAccuracies(c), comboAUCs(c));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%% ===== AGGREGATE OBSERVED METRICS ACROSS COMBINATIONS ===================
fprintf('\nAggregating observed metrics across %d combinations...\n', nCombos);

obsAccuracy         = mean(comboAccuracies);
obsBalancedAccuracy = mean(comboBalancedAccuracies);
obsSensitivity      = mean(comboSensitivities);
obsSpecificity      = mean(comboSpecificities);
obsAUC              = nanmean(comboAUCs);

% Aggregate feature weights: mean across all combinations and folds
allCoeffs = reshape(comboAllCoeffs, [], nFeatures);  % (nCombos*12) x nFeatures
rawWeights = mean(allCoeffs, 1);  % 1 x nFeatures
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

%% ===== COLLECT POOLED POSTERIORS FOR ROC CURVE ==========================
% Accumulate per-observation posteriors across all combinations.
% Each observation may appear in multiple combinations (those that did NOT
% remove its segment), so we average the posteriors across those combos.

allComboPosteriorsFull = NaN(nObservations, nCombos);

for c = 1:nCombos
    removeSegIDs_c = uniqueSegments(removalCombos(c, :));
    keepMask_c     = ~ismember(session_id, removeSegIDs_c);

    X_c   = X(keepMask_c, :);
    y_c   = binaryLabels(keepMask_c);
    sid_c = session_id(keepMask_c);

    remainSegIDs_c = uniqueSegments(setdiff(1:nSegments, removalCombos(c, :)));
    nRemain_c      = numel(remainSegIDs_c);

    posteriors_c = NaN(sum(keepMask_c), 1);

    for k = 1:nRemain_c
        testSID  = remainSegIDs_c(k);
        tMask    = (sid_c == testSID);
        trMask   = ~tMask;

        y_tr     = y_c(trMask);
        X_tr_raw = X_c(trMask, :);
        X_te     = X_c(tMask, :);

        % Same per-fold downsampling as the observed loop
        i0 = find(y_tr == 0);  i1 = find(y_tr == 1);
        n0t = numel(i0);        n1t = numel(i1);
        if n0t > n1t
            keep0 = i0(randperm(n0t, n1t));
            trIdx = sort([keep0; i1]);
        elseif n1t > n0t
            keep1 = i1(randperm(n1t, n0t));
            trIdx = sort([i0; keep1]);
        else
            trIdx = (1:n0t+n1t)';
        end

        mdlR = fitcdiscr(X_tr_raw(trIdx, :), y_tr(trIdx));
        [~, postR] = predict(mdlR, X_te);
        c1idx = find(mdlR.ClassNames == 1);
        if ~isempty(c1idx)
            posteriors_c(tMask) = postR(:, c1idx);
        end
    end

    % Map back into full observation space
    fullPost = NaN(nObservations, 1);
    obsInCombo = find(keepMask_c);
    fullPost(obsInCombo) = posteriors_c;
    allComboPosteriorsFull(:, c) = fullPost;
end

% Average posteriors across combinations (ignoring combos that excluded each obs)
pooledPosteriors = nanmean(allComboPosteriorsFull, 2);  % NaN where obs never appeared
validObsMask     = ~isnan(pooledPosteriors);

% Save into results
results.observed.posteriors   = pooledPosteriors;
results.observed.trueLabels   = binaryLabels;
results.observed.validObsMask = validObsMask;

fprintf('Pooled posteriors collected for %d/%d observations.\n', ...
    sum(validObsMask), nObservations);

%% ===== PERMUTATION TESTING ==============================================
fprintf('\nRunning %d permutations (segment-level label shuffling, combo+downsample)...\n', nPermutations);

% Broadcast variables for parfor
X_broadcast = X;
sid_broadcast = session_id;
uSeg_broadcast = uniqueSegments;
nSeg_broadcast = nSegments;
nFeat_broadcast = nFeatures;
segIndices_broadcast = segIndices;
segBinaryLabels = segmentLabels;
allGroups_bc = allGroups;

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

    % Re-identify majority class under permuted labels
    nC0_p = sum(shuffledSegLabels == 0);
    nC1_p = sum(shuffledSegLabels == 1);
    if nC0_p > nC1_p
        majLabel_p = 0;
    else
        majLabel_p = 1;
    end

    majSegIdx_p = find(shuffledSegLabels == majLabel_p);

    % Re-enumerate removal combos for this permutation
    permRemovalCombos = zeros(0, 3);
    for g = 1:numel(allGroups_bc)
        groupIDs = allGroups_bc{g};
        groupIdx = find(ismember(uSeg_broadcast, groupIDs));
        if all(ismember(groupIdx, majSegIdx_p))
            permRemovalCombos = [permRemovalCombos; groupIdx(:)'];
        end
    end
    nCombos_p = size(permRemovalCombos, 1);

    % Handle edge case: if no valid removal group found
    if nCombos_p == 0
        permBalancedAccuracies(p) = NaN;
        permAUCs(p) = NaN;
        continue;
    end

    comboBAs_p   = zeros(nCombos_p, 1);
    comboAUCs_p  = zeros(nCombos_p, 1);

    for cp = 1:nCombos_p
        removeIdx_p = permRemovalCombos(cp, :);
        removeIDs_p = uSeg_broadcast(removeIdx_p);

        keepMask_p = ~ismember(sid_broadcast, removeIDs_p);
        X_c = X_broadcast(keepMask_p, :);
        y_c = permLabels(keepMask_p);
        sid_c = sid_broadcast(keepMask_p);

        remainSegs_p = setdiff(uSeg_broadcast, removeIDs_p);
        nRemain_p = numel(remainSegs_p);

        predAll_p = zeros(size(y_c));
        postAll_p = zeros(size(y_c));

        for kk = 1:nRemain_p
            tMask = (sid_c == remainSegs_p(kk));
            trMask = ~tMask;

            y_tr = y_c(trMask);
            X_tr = X_c(trMask, :);
            X_te = X_c(tMask, :);

            % Per-fold downsampling on training data
            i0 = find(y_tr == 0);
            i1 = find(y_tr == 1);
            n0t = numel(i0);
            n1t = numel(i1);

            if n0t > n1t && n1t > 0
                keep0 = i0(randperm(n0t, n1t));
                trIdx = sort([keep0; i1]);
            elseif n1t > n0t && n0t > 0
                keep1 = i1(randperm(n1t, n0t));
                trIdx = sort([i0; keep1]);
            else
                trIdx = (1:numel(y_tr))';
            end

            X_tr_ds = X_tr(trIdx, :);
            y_tr_ds = y_tr(trIdx);

            % Train plain LDA and predict
            mdlP = fitcdiscr(X_tr_ds, y_tr_ds);
            [pred_p, post_p] = predict(mdlP, X_te);

            c1 = find(mdlP.ClassNames == 1);
            if isempty(c1)
                postAll_p(tMask) = 0;
            else
                postAll_p(tMask) = post_p(:, c1);
            end
            predAll_p(tMask) = pred_p;
        end

        % Compute permutation metrics for this combo
        tp = sum(predAll_p == 1 & y_c == 1);
        tn = sum(predAll_p == 0 & y_c == 0);
        fp = sum(predAll_p == 1 & y_c == 0);
        fn = sum(predAll_p == 0 & y_c == 1);
        sens_pp = tp / max(tp + fn, 1);
        spec_pp = tn / max(tn + fp, 1);
        comboBAs_p(cp) = (sens_pp + spec_pp) / 2;

        try
            [~, ~, ~, auc_pp] = perfcurve(y_c, postAll_p, 1);
            comboAUCs_p(cp) = auc_pp;
        catch
            comboAUCs_p(cp) = NaN;
        end
    end

    permBalancedAccuracies(p) = nanmean(comboBAs_p);
    permAUCs(p) = nanmean(comboAUCs_p);

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

% Global observed results (aggregated across combinations)
results.observed.accuracy          = obsAccuracy;
results.observed.balancedAccuracy  = obsBalancedAccuracy;
results.observed.sensitivity       = obsSensitivity;
results.observed.specificity       = obsSpecificity;
results.observed.AUC               = obsAUC;
results.observed.featureWeights.raw    = rawWeights;
results.observed.featureWeights.scaled = scaledWeights;

% Per-combination results
for c = nCombos:-1:1
    results.combinations(c).removedSegments    = uniqueSegments(removalCombos(c, :));
    results.combinations(c).accuracy           = comboAccuracies(c);
    results.combinations(c).balancedAccuracy   = comboBalancedAccuracies(c);
    results.combinations(c).sensitivity        = comboSensitivities(c);
    results.combinations(c).specificity        = comboSpecificities(c);
    results.combinations(c).AUC                = comboAUCs(c);
    results.combinations(c).foldCoefficients   = squeeze(comboAllCoeffs(c, :, :));
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
results.info.nCombinations     = nCombos;
results.info.removalGroups     = allGroups;
results.info.removalCombos     = removalCombos;
results.info.dateRun           = datestr(now, 'yyyy-mm-dd HH:MM:SS');

%% ===== SAVE RESULTS =====================================================
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
resultsFilename = sprintf('decoding_results_%s.mat', timestamp);
resultsPath = fullfile(dataFolder, resultsFilename);
save(resultsPath, 'results');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', resultsPath);
fprintf('  Combinations:        %d\n', nCombos);
fprintf('  Accuracy:            %.4f\n', results.observed.accuracy);
fprintf('  Balanced Accuracy:   %.4f (p = %.4f)\n', ...
    results.observed.balancedAccuracy, pValue_balancedAccuracy);
fprintf('  AUC:                 %.4f (p = %.4f)\n', ...
    results.observed.AUC, pValue_AUC);
fprintf('  Sensitivity:         %.4f\n', results.observed.sensitivity);
fprintf('  Specificity:         %.4f\n', results.observed.specificity);
fprintf('=============================\n');
