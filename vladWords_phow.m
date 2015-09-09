function encoder = vladWords_phow(images, numWords, pca)

samplesPerWord = 1000;
numSamples = samplesPerWord * numWords;

numImages = numel(images);
numDescrsPerImage = ceil(numSamples) / numImages;
X = cell(numImages,1);

for ii = 1:numImages
    dat = load(sprintf('phow_desc_new/%s_phow.mat', images{ii}));
    
    nzind= find(dat.fr(3,:) > 0.01 & sum(dat.F,1) > 0);
    rind  = randperm(numel(nzind),min(numDescrsPerImage,numel(nzind)));
    X{ii}  = dat.F(:,nzind(rind));
end
%- Parameters -%

descrs  = single(cat(2,X{:}));

% doing PCA
if pca.numPcaDimensions < inf || pca.whitening
    encoder.projectionCenter = mean(descrs,2) ;
    x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
    X = x*x' / size(x,2) ;
    [V,D] = eig(X) ;
    d = diag(D) ;
    [d,perm] = sort(d,'descend') ;
    d = d + pca.whiteningRegul * max(d) ;
    m = min(pca.numPcaDimensions, size(descrs,1)) ;
    V = V(:,perm) ;
    if pca.whitening
        encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
    else
        encoder.projection = V(:,1:m)' ;
    end
    clear X V D d ;
else
    encoder.projection = 1 ;
    encoder.projectionCenter = 0 ;
end
descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;

if pca.renormalize
    descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
end

randind = randperm(size(descrs,2), numWords);

[encoder.words,~] = mykmeans(descrs, descrs(:,randind));
encoder.kdtree = vl_kdtreebuild(encoder.words, 'numTrees', 2) ;
encoder.numWords = numWords;