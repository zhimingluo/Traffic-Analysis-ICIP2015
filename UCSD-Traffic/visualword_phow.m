function visualword_phow(NUM_OF_WORDS, using_roi, select_frames)

load('trafficdb\imageMaster.mat');
load('trafficdb\EvalSet.mat');

%- Parameters -%

mkdir('visualwords_ROI');
mkdir('visualwords');

for rr = 1:4
    %get the training list
    ntrainsample = numel(trainind{rr}) * length(select_frames);
    
    num_per_file = round(1e5/ntrainsample);
    X = cell(ntrainsample,1);
    index = trainind{rr};
    
    count = 1;
    for ii = 1:numel(index)
        for kk = 1:length(select_frames)
            if using_roi == 1
                dat   = load(['PHOW_ROI\' num2str(index(ii)) '\' num2str(select_frames(kk)) '.mat']);
            else
                dat   = load(['PHOW\' num2str(index(ii)) '\' num2str(select_frames(kk)) '.mat']);
            end
            nzind= find(dat.fr(3,:) > 0.01 & sum(dat.F,1) > 0);
            rind  = randperm(numel(nzind),min(num_per_file,numel(nzind)));
            X{count}  = dat.F(:,nzind(rind));
            count = count + 1;
        end
    end
    
    X  = single(cat(2,X{:}));
    
    for nbin = NUM_OF_WORDS
        fprintf('\nK-means: %4d-clusters\n', nbin);
        randind = randperm(size(X,2), nbin);
        [words, ~] = mykmeans(X, X(:,randind));
        if using_roi == 1
            save(['visualwords_ROI/'  num2str(nbin) '_' num2str(rr) '.mat'], 'words');
        else
            save(['visualwords/'  num2str(nbin) '_' num2str(rr) '.mat'], 'words');
        end
    end
end
