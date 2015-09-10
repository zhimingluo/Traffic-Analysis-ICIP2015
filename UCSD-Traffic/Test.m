using_roi = 0;
NUM_OF_WORDS = 64;
select_frames = [4,10,16,22,28,34,40,46];

%%convert video to frames
%Video2Image(using_roi);

%%compute the PHOW features for the select frames of each video
%calc_phow(using_roi, select_frames);

%%compute the codebook using the training set's select frames
%visualword_phow(NUM_OF_WORDS, using_roi, select_frames);

%%compute the BoW features of the select frames
Image2Feature(using_roi, NUM_OF_WORDS, select_frames);

%%Get the labels of each video
load('trafficdb\ImageMaster.mat');
label_of_videos = zeros(1,numel(imagemaster));

for i = 1:numel(imagemaster)
    switch imagemaster{i}.class
        case 'heavy'
            label_of_videos(i) = 1;
        case 'medium'
            label_of_videos(i) = 2;
        case 'light'
            label_of_videos(i) = 3;
    end
end

%%Traning SVM and Classify
% Training Set:
%   The label of each select frame is copied its video's label
% Test Set:
%   Video label is the majority label of its select frames.
svm.C = 1;
svm.kernel = 'hik'; %linear, hell, chi2, hik
TrainSVMClassiy(label_of_videos, NUM_OF_WORDS, using_roi, svm)