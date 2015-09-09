function Result = SVM_Classify(type, numWords, svm)

for rr = 1:10  
    load(['Train_Test_Split/cv_data_test' num2str(rr) '.mat']);
    load(sprintf('Coding/%s_%d_%d.mat', type, numWords, rr));
    
    ncls = numel(classes);
    
    for ii = 1:numel(trainflist)
        tr_index(ii) = str2num(trainflist{ii});
    end
    
    for ii = 1:numel(testflist)
        ts_index(ii) = str2num(testflist{ii});
    end
    
    switch svm.kernel
        case 'linear'
        case 'hell'
            imgFeatures = sign(imgFeatures) .* sqrt(abs(imgFeatures)) ;
        case 'chi2'
            imgFeatures = vl_homkermap(imgFeatures,1,'kchi2') ;
        case 'hik'
            imgFeatures = vl_homkermap(imgFeatures,1,'kinters');
        otherwise
            assert(false) ;
    end
    imgFeatures = snorm(imgFeatures);
    
    tr_fea = double(imgFeatures(:, tr_index));
    [~,tr_label] = max(trainID,[],2);
    
    ts_fea = double(imgFeatures(:,ts_index));
    
    [~,ts_label] = max(testID,[],2);
    
    options = ['-c ' num2str(svm.c)];
    model = train(double(tr_label), sparse(tr_fea)', options);
    
    curr_C = predict(ts_label, sparse(ts_fea)', model);
    
    %compute mean AP and confusion Matrix
    confmat = full(sparse(ts_label, curr_C, 1, ncls, ncls));
    confmat = bsxfun(@times, confmat, 1./sum(confmat,2));
    macc = mean(diag(confmat));
    res(rr).macc = macc;
    res(rr).confmat = confmat;
    res(rr).testID = [ts_label, curr_C];
end

acc = zeros(1,numel(res));
confuse = zeros(ncls:ncls);

for ii = 1:numel(res)
    acc(ii) = res(ii).macc;
    confuse = confuse + res(ii).confmat;
end
mm = mean(acc);
confuseM = confuse / numel(res);

Result.res = res;
Result.acc = mm;
Result.confuseM = confuseM;
