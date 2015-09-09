function DeepFeature(Model)

run(fullfile('D:/Toolbox/matconvnet-1.0-beta7', '/matlab/vl_setupnn.m')) ;

net = load(fullfile('D:\Toolbox\matconvnet-pretrain-model',Model));

imgFeatures = cell(400,1);

ReLU = 1;

for ii = 1:400
    imageName = sprintf('images/%06d.jpg',ii);
    fprintf('%s\n',imageName);
    im = imread(imageName) ;
     
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
    im_ = im_ - net.normalization.averageImage ;
       
    % run the CNN
    res = vl_simplenn(net, im_) ;
    if ReLU == 1
        imgFeature = squeeze(gather(res(end-2).x));
    else
        imgFeature = squeeze(gather(res(end-3).x));
    end
    imgFeatures{ii} = snorm(imgFeature);
end

imgFeatures = cat(2,imgFeatures{:});

save(sprintf('Deep_Feature/Feature_%s.mat',Model),'imgFeatures');
