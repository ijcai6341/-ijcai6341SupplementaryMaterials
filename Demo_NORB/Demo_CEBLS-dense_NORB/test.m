
%%%%%%%%%%load the dataset NORB dataset%%%%%%%%%%%%%%%%%%%%
clear; 
warning off all;
format compact;
load norb;

%%%%%%%%%%%%%%%the samples from the data are normalized and the lable data
%%%%%%%%%%%%%%%train_y and test_y are reset as N*C matrices%%%%%%%%%%%%%%
train_x = double(train_x/255);
train_y = double(train_y);
% test_x = double(train_x/255);
% test_y = double(train_y);
test_x = double(test_x/255);
test_y = double(test_y);
train_y=(train_y-1)*2+1;
test_y=(test_y-1)*2+1;
assert(isfloat(train_x), 'train_x must be a float');
assert(all(train_x(:)>=0) && all(train_x(:)<=1), 'all data in train_x must be in [0:1]');
assert(isfloat(test_x), 'test_x must be a float');
assert(all(test_x(:)>=0) && all(test_x(:)<=1), 'all data in test_x must be in [0:1]');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%The preprossing for the norb data: ZCA whiten%%%%%%%
[Train_x, Test_x]=pre_zca(train_x,test_x);
train_x=Train_x;test_x=Test_x;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%This is the model of CEBLS-dense%%%%%%

C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=100;%feature nodes  per window
N2=21;% number of windows of feature nodes
N33=800;% number of enhancement nodes
N44 =10;
N0=4;
epochs=10;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
N1=N11; N3=N33;  
for j=1:epochs    
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = CEBLS_dense_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,N44);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end
save ( ['norb_result_oneshot_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%