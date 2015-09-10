function Image2Feature(using_roi, num_words, select_frames)
% Compute the BoW features for the select frames of each video

load('trafficdb\ImageMaster.mat');

mkdir(['Features\' num2str(num_words)]);

videoNums = numel(imagemaster);

% Different training configurations, using specific codebook.
for rr = 1:4
    
    if using_roi == 1
        mkdir(['Features_ROI\' num2str(num_words) '\' num2str(rr)]);  
        load(['visualwords_ROI\' num2str(num_words) '_' num2str(rr) '.mat']);
    else
        mkdir(['Features\' num2str(num_words) '\' num2str(rr)]);  
        load(['visualwords\' num2str(num_words) '_' num2str(rr) '.mat']);
    end
     
    for ii = 1:videoNums
        features = cell(numel(select_frames),1);
        
        for jj = 1:numel(select_frames)
            
            if using_roi == 1
                PhowPath = ['PHOW_ROI\' num2str(ii) '\' num2str(select_frames(jj)) '.mat'];
            else
                PhowPath = ['PHOW\' num2str(ii) '\' num2str(select_frames(jj)) '.mat'];
            end
            load(PhowPath);
            
            [inds, ~] = myknn(single(F), words, 1);
            feature = zeros( num_words, 1) ;
            feature = vl_binsum(feature, ones(size(inds)), inds) ;
            
            features{jj} = snorm(feature);
        end
        
        feature = cat(2,features{:});
        
        if using_roi == 1
            save(['Features_ROI\' num2str(num_words) '\' num2str(rr) '\' num2str(ii) '.mat'],'feature');
        else
            save(['Features\' num2str(num_words) '\' num2str(rr) '\' num2str(ii) '.mat'],'feature');
        end
    end
end

