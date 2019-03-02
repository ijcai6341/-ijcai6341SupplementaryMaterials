%%%%%%%%%%%%%%%%%%%%%%%%This is the demo for the bls models including the
%%%%%%%%%%%%%%%%%%%%%%%%proposed incremental learning algorithms. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%load the dataset MNIST dataset%%%%%%%%%%%%%%%%%%%%
clear; 
warning off all;
format compact;
load mnist;

%%%%%%%%%%%%%%%the samples from the data are normalized and the lable data
%%%%%%%%%%%%%%%train_y and test_y are reset as N*C matrices%%%%%%%%%%%%%%
train_x = double(train_x/255);
train_y = double(train_y);
test_x = double(test_x/255);
test_y = double(test_y);
train_y=(train_y-1)*2+1;
test_y=(test_y-1)*2+1;
assert(isfloat(train_x), 'train_x must be a float');
assert(all(train_x(:)>=0) && all(train_x(:)<=1), 'all data in train_x must be in [0:1]');
assert(isfloat(test_x), 'test_x must be a float');
assert(all(test_x(:)>=0) && all(test_x(:)<=1), 'all data in test_x must be in [0:1]');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% disp('Press any key to run the one shot BLS demo');
% pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning sytem with%%%%%%
%%%%%%%%%%%%%%%%%%%%one shot structrue%%%%%%%%%%%%%%%%%%%%%%%%
C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=15;%feature nodes  per window
N0=5;% number of layer of feature nodes
N33=13000;% number of enhancement nodes
epochs=1;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
for j=1:epochs    
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = CFBLS_pyramid_train(train_x,train_y,test_x,test_y,s,C,N1,N3,N0);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end
save ( ['mnist_result_oneshot_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
