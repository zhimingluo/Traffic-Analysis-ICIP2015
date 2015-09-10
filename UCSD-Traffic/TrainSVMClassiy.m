function TrainSVMClassiy(videoLabel, num_words, using_roi, svm)

load('trafficdb\EvalSet.mat');
load('trafficdb\ImageMaster.mat');

numOfVideos = numel(imagemaster);

MyLabels = zeros(1,numOfVideos);

for rr = 1:4
    for ii = 1:numOfVideos
        if using_roi == 1
            A = load(['Features_ROI\' num2str(num_words) '\' num2str(rr) '\' num2str(ii) '.mat']);
        else
            A = load(['Features\' num2str(num_words) '\' num2str(rr) '\' num2str(ii) '.mat']);
        end
        Fea{ii} = A.feature;
    end
    
    index = trainind{rr};
    select_frames = size(Fea{1},2);
    labels = repmat(videoLabel, select_frames ,1);
    
    tr_fea = Fea(index);
    tr_label = labels(:,index);
    
    tr_fea = double(cat( 2, tr_fea{:}));
    tr_label = tr_label(:);
    
    ts_index = testind{rr};
    ts_fea = Fea(ts_index);
    ts_label = labels(:,ts_index);
    ts_fea = double(cat( 2, ts_fea{:}));
    ts_label = ts_label(:);
    
    fea = double(cat( 2, Fea{:}));
    label = labels(:);
    
    switch svm.kernel
        case 'linear'
        case 'hell'
            tr_fea = sign(tr_fea) .* sqrt(abs(tr_fea)) ;
            ts_fea = sign(ts_fea) .* sqrt(abs(ts_fea)) ;
        case 'chi2'
            tr_fea = vl_homkermap(tr_fea,1,'kchi2') ;
            ts_fea = vl_homkermap(ts_fea,1,'kchi2') ;
        case 'hik'
            tr_fea = vl_homkermap(tr_fea,1,'kinters');
            ts_fea = vl_homkermap(ts_fea,1,'kinters');
        otherwise
            assert(false) ;
    end
    tr_fea = bsxfun(@times, tr_fea, 1./sqrt(sum(tr_fea.^2))) ;
    ts_fea = bsxfun(@times, ts_fea, 1./sqrt(sum(ts_fea.^2))) ;
    
    options = ['-q -c ' num2str(svm.C)];
    model = train(double(tr_label), sparse(tr_fea)', options);
    
    [curr_C] = predict(ts_label, sparse(ts_fea)', model);
    
    curr_C = reshape(curr_C,select_frames,[]);
    
    % major voting through the select frames
    accLabel = [];
    accLabel(1,:) = sum( curr_C == 1);
    accLabel(2,:) = sum( curr_C == 2);
    accLabel(3,:) = sum( curr_C == 3);
    [~,preLabel] = max(accLabel, [], 1);
    
    MyLabels(ts_index) = preLabel;
    
    acc(rr) = sum(preLabel == videoLabel(ts_index)) / numel(ts_index);

end

fprintf('The whole classification accuracy:%.2f%%\n',sum(MyLabels == videoLabel) / numOfVideos * 100 );