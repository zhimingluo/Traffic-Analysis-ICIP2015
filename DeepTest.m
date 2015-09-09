%Model = 'imagenet-caffe-ref';
Model = 'imagenet-vgg-f';

FeaturePath = sprintf('Deep_Feature/Feature_%s.mat',Model);

if ~exist(FeaturePath,'file')
    DeepFeature(Model);
end

load(FeaturePath);

%kernel = ['linear', 'hell', 'chi2', 'hik'];

svm.kernel = 'hik';
svm.c = 1;

addpath('D:\Toolbox\liblinear-1.94\matlab');

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
imgFeatures = bsxfun(@times, imgFeatures, 1./sqrt(sum(imgFeatures.^2))) ;

for rr = 1:10
    
    load(['Train_Test_Split\cv_data_test' num2str(rr) '.mat']);
    
    ncls = numel(classes);
    
    for ii = 1:numel(trainflist)
        tr_index(ii) = str2num(trainflist{ii});
    end
    
    for ii = 1:numel(testflist)
        ts_index(ii) = str2num(testflist{ii});
    end
    
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
    result(rr).macc = macc;
    result(rr).confmat = confmat;
    result(rr).testID = [ts_label, curr_C];
end

acc = zeros(1,numel(result));
confuse = zeros(ncls:ncls);

for ii = 1:numel(result)
    acc(ii) = result(ii).macc;
    confuse = confuse + result(ii).confmat;
end

mm = mean(acc);
confuseM = confuse / numel(result);

Result.res = result;
Result.acc = mm;
Result.confuseM = confuseM;
fprintf('\n\n------------------------------\n');
fprintf('Mean Accuracy: %.2f%%\n',Result.acc*100);
fprintf('------------------------------\n');
% draw confusion matrice
addpath('ConfusionMatrices/');
figure();
draw_cm(Result.confuseM,classes, ncls);
