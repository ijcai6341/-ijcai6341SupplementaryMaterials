function y=get_labels(x)
y=zeros(size(x,1),int32(max(x)));
t=0;
for i=1:size(x,1)
    t=int32(abs(x(i)));
    y(i,t)=1;
end