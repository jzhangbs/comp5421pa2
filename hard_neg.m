% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function features_hard_neg = .... 
    hard_neg(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
num_test_scenes = length(test_scenes);
p = gcp();
features_hard_neg = [];

for i = 1:num_test_scenes
    f(i) = parfeval(p, @hard_neg_core, 1, fullfile( test_scn_path, test_scenes(i).name ), w, b, feature_params);
end

for i = 1:num_test_scenes
    if (mod(i,20)==0)
        fprintf('%d/%d, %d\n', i, num_test_scenes, length(features_hard_neg(:,1)));
    end
    [idx, feat] = fetchNext(f);
    features_hard_neg = [features_hard_neg; feat];
end

function feat = hard_neg_core(filename, w, b, feature_params)

img = imread(filename);
img = single(img)/255;
if(size(img,3) > 1)
    img = rgb2gray(img);
end

feat = [];
thresh = 0.2;
for scale = 1:1
    img_rs = imresize(img, scale, 'bicubic');
    hog = vl_hog(img_rs, feature_params.hog_cell_size);
    win_cell_size = feature_params.template_size / feature_params.hog_cell_size;
    for im_i = 1:(size(hog, 1)-win_cell_size+1)
        for im_j = 1:(size(hog, 2)-win_cell_size+1)
            win = hog(im_i:im_i+win_cell_size-1, im_j:im_j+win_cell_size-1, :);
            imshow(vl_hog('render', win));
            win_flat = reshape(win, 1, []);
            new_conf = win_flat*w+b;
            new_conf = new_conf(1,1);
            if (new_conf > thresh)
                feat = [feat; win_flat];
            end
        end
    end
end