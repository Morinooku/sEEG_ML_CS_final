clear; clc;

%% Define parameters
channel = [3:10 22 23 11:13 15:21 24 25:32 37:38 47 49:63 64:68 70:81 82:88 90:93 94:105 106:117 118:125 127:133 134:148];%%%%%%%%%%%%%%%%%%%%%%%%
frequency_band = {[1 4], [4 8], [8 12], [12 32], [32 45; 55 60], [60 90]};
band_names = {'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma'};
decodingscore_list = [7 7 7 6 6];%%%%%%%%%%%%%%%%%%%%%%%%
subname = '07_FHX_SF-MPQ-';%%%%%%%%%%%%%%%%%%%%%%%%
num_sessions = 5;
data_path = 'E:\8LFP\6PSD\07_FHX\';%%%%%%%%%%%%%%%%%%%%%%%%
save_path = 'E:\8LFP\7MVA\07_FHX\allsite\';%%%%%%%%%%%%%%%%%%%%%%%%
file_suffixes = {'_before', '_after'};%%%%%%%%%%%%%%%%%%%%%%%%

%% Calculate dimensions
num_sites = length(channel);
num_bands = length(frequency_band);

%% Initialize output variables for each band
inputdata_by_band   = cell(num_bands, 1);
decodescore_by_band = cell(num_bands, 1);
sessionIds_by_band  = cell(num_bands, 1);
segmentIds_by_band  = cell(num_bands, 1);

for band = 1:num_bands
    inputdata_by_band{band}   = [];
    decodescore_by_band{band} = [];
    sessionIds_by_band{band}  = [];
    segmentIds_by_band{band}  = [];
end

%% Loop through sessions and recordings
for session = 1:num_sessions
    for suffix_idx = 1:length(file_suffixes)
        suffix = file_suffixes{suffix_idx};
        
        % Construct filename
        filename = fullfile(data_path, sprintf([subname '%d%s.mat'], session, suffix));
        
        if ~exist(filename, 'file')
            warning('File not found: %s. Skipping...', filename);
            continue;
        end
        
        % Load data
        fprintf('Loading: %s\n', filename);
        data = load(filename);
        f        = data.f(:);
        PSD_data = data.PSD_data;
        
        PSD_selected = PSD_data(channel, :, :);
        num_epochs   = size(PSD_selected, 3);
        fprintf('  - Found %d epochs\n', num_epochs);
        
        % Global segment index: 1–15 (non-overlapping across sessions)
        segment_global_id = (session - 1) * length(file_suffixes) + suffix_idx;
        
        % Pre-allocate feature matrices for this recording
        epoch_features_by_band = cell(num_bands, 1);
        for band = 1:num_bands
            epoch_features_by_band{band} = zeros(num_epochs, num_sites);
        end
        
        %% Extract features for each epoch
        for epoch = 1:num_epochs
            psd_epoch = PSD_selected(:, :, epoch);
            
            for band = 1:num_bands
                band_range = frequency_band{band};
                
                freq_idx = false(size(f));
                for row = 1:size(band_range, 1)
                    freq_idx = freq_idx | (f >= band_range(row, 1) & f <= band_range(row, 2));
                end
                
                band_power = 10 * log10(mean(psd_epoch(:, freq_idx), 2));
                epoch_features_by_band{band}(epoch, :) = band_power';
            end
        end
        
        % Append to output for each band
        for band = 1:num_bands
            inputdata_by_band{band}   = [inputdata_by_band{band};   epoch_features_by_band{band}];
            decodescore_by_band{band} = [decodescore_by_band{band}; repmat(decodingscore_list(session), num_epochs, 1)];
            sessionIds_by_band{band}  = [sessionIds_by_band{band};  repmat(session,            num_epochs, 1)];
            segmentIds_by_band{band}  = [segmentIds_by_band{band};  repmat(segment_global_id,  num_epochs, 1)];
        end
    end
end

%% Display summary
fprintf('\n===== Data Organization Complete =====\n');
for band = 1:num_bands
    num_epochs_band = size(inputdata_by_band{band}, 1);
    fprintf('Band %d (%s): %d epochs, %d sites\n', band, band_names{band}, num_epochs_band, num_sites);
end

%% Convert and save each band
fprintf('\n===== Saving Band-Specific Files =====\n');
for band = 1:num_bands
    num_epochs_band = size(inputdata_by_band{band}, 1);
    
    % Build cell array for inputData
    inputData = cell(num_epochs_band, num_sites);
    for epoch = 1:num_epochs_band
        for site = 1:num_sites
            inputData{epoch, site} = inputdata_by_band{band}(epoch, site);
        end
    end
    
    % Rename to final variable names
    decodeScore = decodescore_by_band{band};          % pain score per epoch
    session_id  = segmentIds_by_band{band};           % 1–15 sequential segment index
    
    % --- MODIFIED: create a dedicated subfolder for each band ---
    band_name    = band_names{band};
    file_name    = sprintf('decoding_data_%s', band_name);   % shared folder & file stem
    band_folder  = fullfile(save_path, file_name);           % folder = file stem

    if ~exist(band_folder, 'dir')
        mkdir(band_folder);
    end

    filename_band = fullfile(band_folder, sprintf('%s.mat', file_name));
    % ------------------------------------------------------------

    save(filename_band, 'channel', 'inputData', 'decodeScore', 'session_id');
    
    fprintf('Saved: %s  (inputData: [%d x %d] cell, decodeScore: [%d x 1])\n', ...
        filename_band, size(inputData, 1), size(inputData, 2), length(decodeScore));
    
    fprintf('  Epochs per pain score:\n');
    unique_scores = unique(decodeScore);
    for i = 1:length(unique_scores)
        count = sum(decodeScore == unique_scores(i));
        fprintf('    Score %d: %d epochs\n', unique_scores(i), count);
    end
    
    fprintf('  Segment ID range: %d – %d\n', min(session_id), max(session_id));
end

fprintf('\nAll band-specific files saved successfully!\n');