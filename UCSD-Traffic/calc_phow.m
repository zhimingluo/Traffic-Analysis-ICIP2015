function calc_phow(using_roi, select_frames)

load('trafficdb\ImageMaster.mat');
videoNum = numel(imagemaster);

%- Parameters -%
step = 4;
patchsizes = [16 24 32];

for ii = 1:videoNum
    
    if using_roi == 1 
        PATH_TO_IMAGES = ['Images_ROI\', num2str(ii)]; 
        PATH_TO_Features = ['PHOW_ROI\', num2str(ii) '\'];
    else
        PATH_TO_IMAGES = ['Images\', num2str(ii)]; 
        PATH_TO_Features = ['PHOW\', num2str(ii) '\'];
    end
    
    mkdir(PATH_TO_Features);
    
    for i = 1:length(select_frames)
        
        img = imread(sprintf('%s/%d.bmp',PATH_TO_IMAGES, select_frames(i)));
        
        fprintf('%s\\%d.bmp\n',PATH_TO_IMAGES, select_frames(i));
        
        if ndims(img) == 3,
            img = im2single(rgb2gray(img));
        else
            img = im2single(img);
        end;
        [fr F] = vl_phow(img, 'Sizes',patchsizes/4, 'Step',step);
        
        save([PATH_TO_Features, num2str(select_frames(i)), '.mat'], 'fr','F');
    end
    
end

