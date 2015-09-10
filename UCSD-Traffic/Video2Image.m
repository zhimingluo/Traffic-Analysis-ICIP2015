function Video2Image(using_roi)

load('trafficdb\ImageMaster.mat');
NumsOfVideos = numel(imagemaster);


ROI_x1 = 45;
ROI_y1 = 120;
ROI_x2 = ROI_x1+180-1;
ROI_y2 = ROI_y1+180-1;

mkdir('Images');
mkdir('Images_ROI');

for ii = 1:NumsOfVideos
    
    videoObj = VideoReader(['trafficdb\video\' imagemaster{ii}.root '.avi']);
    Frames = videoObj.NumberOfFrames;
    
    if using_roi
        mkdir(['Images_ROI\' num2str(ii)]);
    else
        mkdir(['Images\' num2str(ii)]);
    end
    
    fprintf('%d: %s\n',ii,imagemaster{ii}.class);
    
    for jj = 1:Frames
        Image = read(videoObj,jj);
        if using_roi
            ROI = Image(ROI_x1:ROI_x2,ROI_y1:ROI_y2,:);
            imwrite(ROI, ['Images_ROI\' num2str(ii) '\' num2str(jj) '.bmp']);
        else
            imwrite(Image, ['Images\' num2str(ii) '\' num2str(jj) '.bmp']);
        end
    end
end