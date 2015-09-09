function [ train,test ] = equalcvpartition( label,count )
%EQUALCVPARTITION Summary of this function goes here
%Detailed explanation goes here

test=0*label;
labelvec=unique(label);
ss=0;
for i=1:length(labelvec)
    s=nnz(label==labelvec(i));
    idx=randperm(s);
    test(ss+idx(1:count))=1;
    ss=ss+s;
end
test=test==1;
train=~test;

end