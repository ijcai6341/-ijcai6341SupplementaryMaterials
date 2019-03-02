load('extra_32x32.mat');
filename='test_32x32.mat';
file=matfile(filename);

num_points = size(X,4);%X is (X,Y,RGB,index)
num = size(file.X,4);
%size(X,3)
%crop = [3 30 3 30];
test = file.X;
% new_X = zeros((crop(2)-crop(1)+1),(crop(4)-crop(3)+1), num_points);
%nu=size(X,4)
train_x = zeros(num_points,3072);
train_y=get_labels(y);
test_x=zeros(num,3072);
test_y=get_labels(file.y);

%fprintf('Start preprocessing data...\n');
%
% for i= 1:num
%     im = test(:,:,:,i);
%     im=reshape(im,[1,3072]);
%     test_x(i,:)=im;
% end

 for i = 1:num_points
% 
%     
% 
%     %fprintf('Processing image %i/%i...\n', i, num_points);
% 
     im = X(:,:,:,i);
     im=reshape(im,[1,3072]);
     train_x(i,:)=im;
% 
%     %im = rgb2gray(im); 
% 
%     %im = im(crop(1):crop(2),crop(3):crop(4)); % only keep center
% 
%     %im = im2double(im);
% 
%     
% 
%     %new_X(:,:,i) = im; %? 576*num_points????????
% 
%     
% 
%     
% 
%     % Perform ZCA whitening 
% 
%     %[patches_whitened,~,~,~] = whiten(patches, 0.0001);
% 
%     
% 
%     % Save patch data for current patch to cell array
% 
%     %patchData{i} = patches_whitened;
% 
end
train_x=uint8(train_x);
% 

% split_point = 70000;
% 
% seq = randperm(num_points);
% 
% images = new_X(:,:,seq(1:split_point));     %train set data 
% 
% labels = y(seq(1:split_point));           %train set label

save ('svhn_extra', 'train_x','train_y','test_x','test_y');

% images = new_X;