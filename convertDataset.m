PATH_TO_DATASET = 'Motorway';
ROOTPATH = './';
PATH_TO_IMAGES = fullfile(ROOTPATH, 'images');
PATH_TO_TrainTest = fullfile(ROOTPATH, 'Train_Test_Split');

mkdir(PATH_TO_IMAGES);
mkdir(PATH_TO_TrainTest);

dlist = dir(PATH_TO_DATASET);
dlist(1:2) = [];

ncls = numel(dlist);

%- Copy and rename image files -%
count = 1;
for i = 1:numel(dlist)
    flist = dir(fullfile(PATH_TO_DATASET, dlist(i).name, '*.jpg'));
    for j = 1:numel(flist)
        fname = sprintf('%06d',count);
        copyfile(fullfile(PATH_TO_DATASET,dlist(i).name,flist(j).name), sprintf('%s/images/%s.jpg', ROOTPATH, fname));
        ID(count) = i;
        names{count} = fname;
        count = count + 1;
    end
end

classes = {dlist.name};

numTrain = 50;
%- CV data -%
for cvi = 1:10
    rng(cvi);
    %rand('state',cvi)
    [train,test]=equalcvpartition(ID,numTrain);
    %CV = equalcvpartition(ID, 100);
    CV = {};
    CV.test = test;
    CV.training = train;
    
    trainflist = names(CV.test);
    trainID    = full(sparse(1:nnz(CV.test), ID(CV.test), 2, nnz(CV.test),ncls)) - 1;
    vwtrainflist = trainflist;
    vwtrainID    = trainID;
    testflist = names(CV.training);
    testID    = full(sparse(1:nnz(CV.training), ID(CV.training), 2, nnz(CV.training),ncls)) - 1;
    save(fullfile(PATH_TO_TrainTest, sprintf('cv_data_test%d.mat',cvi)), ...
        'classes','trainflist','trainID','vwtrainflist','vwtrainID','testflist','testID');
end
