function Image2Feature(type, numWords, pca)

flist = dir('phow_desc_new/*.mat');

for rr = 1:10
    
    if strcmp(type,'LLC')==0
        load(sprintf('visualwords/%s_%d_%d.mat',type, numWords, rr));
    else
        load(sprintf('visualwords/VQ_%d_%d.mat', numWords, rr));
    end
    
    imgFeatures = cell(numel(flist),1);
    
    for ii = 1:numel(flist)
        load(fullfile('phow_desc_new/', flist(ii).name));
        
        fprintf('%s\n',flist(ii).name);
        
        switch type
            case 'VQ'
                [inds, d] = myknn(single(F), words, 1);
                imgFeature = zeros(numWords, 1);
                imgFeature = vl_binsum(imgFeature, ones(size(inds)), inds) ;
            case 'fisher'
                descrs = words.projection * bsxfun(@minus, F, words.projectionCenter) ;
                if pca.renormalize
                    descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
                end
                imgFeature = vl_fisher(descrs, words.means,...
                    words.covariances, words.priors, ...
                    'Improved') ;
            case 'VLAD'
                descrs = words.projection * bsxfun(@minus, F, words.projectionCenter) ;
                [index,distances] = vl_kdtreequery(words.kdtree, words.words, ...
                    descrs, 'MaxComparisons', 15) ;
                assign = zeros(words.numWords, numel(index), 'single') ;
                assign(sub2ind(size(assign), double(index), 1:numel(index))) = 1 ;
                imgFeature = vl_vlad(descrs, words.words, assign, ...
                    'SquareRoot',  'NormalizeComponents') ;
            case 'LLC'
                llc_codes = LLC_coding_appr(words', F');
                % Average pooling
                %hist = max(llc_codes,[],1)';
                imgFeature =  mean(llc_codes, 1)';
        end
        % L1-norm
        %hists{ii} = single(hist / sum(hist)) ;
        % L2-normalization
        imgFeatures{ii} = snorm(imgFeature);
    end
    
    imgFeatures = cat(2,imgFeatures{:});
    save(sprintf('Coding/%s_%d_%d.mat',type, numWords, rr),'imgFeatures');
end



