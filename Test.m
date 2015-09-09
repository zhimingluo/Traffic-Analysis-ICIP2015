numWords = 64;

% VQ, fisher, VLAD, LLC
type = 'fisher';

LLC.knn = 5;
LLC.beta = 1e-4;

%VLAD, fisher pca parameter
pca.numPcaDimensions = 80;
pca.whiteningRegul = 0.01;
pca.whitening = false;
pca.renormalize = true;

trained = 0;

if trained == 0
    % visual words
    mkdir('visualwords');
    %visualWords(type, numWords, pca);
    
    % Coding
    Image2Feature(type, numWords,pca);
end

addpath('D:\Toolbox\liblinear-1.94\matlab');
svm.c = 1;
%SVM
svm.kernel = 'linear';
LinRes = SVM_Classify(type, numWords, svm);
svm.kernel = 'hell';
HellRes = SVM_Classify(type, numWords, svm);
svm.kernel = 'chi2';
Chi2Res = SVM_Classify(type, numWords, svm);
svm.kernel = 'hik';
HIKRes = SVM_Classify(type, numWords, svm);

fprintf('\n\n------------------------------\n');
fprintf('Mean Accuracy\n');
fprintf('Coding Method: %s, %d visual-words\n',type,numWords);
fprintf('Linear Kernel: %.2f%%\n',LinRes.acc*100);
fprintf('  Hell Kernel: %.2f%%\n',HellRes.acc*100);
fprintf('  Chi2 Kernel: %.2f%%\n',Chi2Res.acc*100);
fprintf('   HIK Kernel: %.2f%%\n',HIKRes.acc*100);
fprintf('------------------------------\n');
