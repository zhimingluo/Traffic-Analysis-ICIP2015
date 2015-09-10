function visualWords(type, numWords, pca)

for ii = 1:10
    load(['Train_Test_Split/cv_data_test' num2str(ii) '.mat']); 
     switch type
        case 'VQ'
            words = kmeansWords_phow(trainflist, numWords);
        case 'fisher'
            words = fisherWords_phow(trainflist, numWords, pca);
        case 'LLC'
            % using the same codebook of VQ
            % words = kmeansWords_phow(trainflist, numWords);
        case 'VLAD'
            words = vladWords_phow(trainflist, numWords, pca);
    end

    save(sprintf('visualwords/%s_%d_%d.mat',type, numWords, ii), 'words');
end
