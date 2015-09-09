function computePhowFeature()

PATH_TO_IMAGES = './images';

flist = dir(fullfile(PATH_TO_IMAGES,'*.jpg'));

max_size = 300;

step = 4;
patchsizes = [4 6 8];

opath = fullfile('./','phow_desc_new'); mkdir(opath);

savefeat = @(fname, F,fr,patchsizes,step,Lh,Lw,toc)...
    save(fname, 'F','fr','patchsizes','step','Lh','Lw');

for i = 1:length(flist)
    img = imread(fullfile(PATH_TO_IMAGES,flist(i).name));
    fprintf('%s\n',flist(i).name);
    
    img = normalizeImage(img, max_size);
    
    [im_h, im_w, ~] = size(img);
    
    [F, fr] = computePhow(img, 'Sizes', patchsizes, 'Step', step);
    
    savefeat(fullfile(opath, regexprep(flist(i).name, '.jpg', '_PHOW.mat')), F, fr, patchsizes, step, im_h, im_w);
end

function [feats, frames] = computePhow(img, varargin )

opts.verbose = false;
opts.sizes = [4 6 8 10];
opts.fast = true;
opts.step = 2;
opts.color = 'gray';
opts.contrast_threshold = 0.005;
opts.window_size = 1.5;
opts.magnif = 6;
opts.float_descriptors = false;

opts.rootSift = false;
opts.normalizeSift = false;

opts = vl_argparse(opts, varargin);

[frames, feats] = vl_phow(img, 'Verbose', opts.verbose, ...
    'Sizes', opts.sizes, 'Fast', opts.fast, 'step', opts.step, ...
    'Color', opts.color, 'ContrastThreshold', opts.contrast_threshold, ...
    'WindowSize', opts.window_size, 'Magnif', opts.magnif, ...
    'FloatDescriptors', opts.float_descriptors);

feats = single(feats);
frames = single(frames);

if opts.rootSift
    feats = sqrt(feats);
end

if opts.normalizeSift
    feats = snorm(feats);
end

function im = normalizeImage(im, max_size)

key_size = max(size(im));

if ndims(im) == 3
    im = im2single(rgb2gray(im));
else
    im = im2single(im);
end

if key_size ~= max_size
    ratio = max_size / key_size;
    im = imresize(im, ratio);
end
