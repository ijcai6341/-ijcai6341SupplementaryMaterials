
%%%%%%%%%%load the dataset NORB dataset%%%%%%%%%%%%%%%%%%%%
clear; 
warning off all;
format compact;
load norb;


%%%%%%%%%%%%%%%%%%%%This is the model of SFEBLS-HOG%%%%%%

C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=100;%feature nodes  per window
N2=10;% number of windows of feature nodes
N33=5000;% number of enhancement nodes
epochs=10;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
N1=N11; N3=N33;  
for j=1:epochs    
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = HOG_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end
save ( ['norb_result_oneshot_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%