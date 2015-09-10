function Words = kmeansWords_phow(images, numWords)

samplesPerWord = 1000;

numSamples = samplesPerWord * numWords;

numImages = numel(images);
numDescrsPerImage = ceil( numSamples ) / numImages;

X = cell(numImages,1);

for ii = 1:numImages
    dat = load(sprintf('phow_desc_new/%s_phow.mat', images{ii}));
    
    nzind= find(dat.fr(3,:) > 0.01 & sum(dat.F,1) > 0); 
    rind  = randperm(numel(nzind),min(numDescrsPerImage,numel(nzind)));
    X{ii}  = dat.F(:,nzind(rind));
end
%- Parameters -%

X  = single(cat(2,X{:}));

fprintf('\nK-means: %4d-clusters\n', numWords);
randind = randperm(size(X,2), numWords);
[Words, ~] = mykmeans(X, X(:,randind));
