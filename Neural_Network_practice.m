clc;clear;close all
%% This code demonstrate how to use one layer multi-neuron network to approximate arbitrary function

%% Original Function generation
p = -2:0.01:2;
i = 2;
Gp = 1+sin(i*pi/4.*p);
figure
plot(p,Gp)
hold on

%% Layer neuron Setting

M = size(p,2);
layerneunum = [20,1];
s2 = zeros(layerneunum(2),M);
s1 = zeros(layerneunum(1),M);
a1 = zeros(layerneunum(1),M);
a2 = zeros(layerneunum(2),M);
e = zeros(1,M);
J = zeros(1,M);

W1 = normrnd(0,1,1,layerneunum(1))';
b1 = normrnd(0,1,1,layerneunum(1))';
W2 = normrnd(0,1,1,layerneunum(1));
b2 = 0.2;
iternum = 1000;
alpha = 0.1;

for j = 1:1:iternum
    for i=1:1:M
        a0 = p;
        t = Gp(i);

        a1(:,i) = logsig(W1*a0(i)+b1);
        a2(:,i) = purelin(W2*a1(:,i)+b2);
        e(i) = t-a2(:,i);


        df1 = diag((1-a1(:,i)).*(a1(:,i)));
        df2 = 1;
        s2(:,i) = -2*df2*e(i) ;
        s1(:,i) = df1*W2'*s2(i);

    end
    
%%  Update weighting function 
J(j) = 1/M*(e*e');
dW1 = alpha/M*s1*a0';
dW2 = alpha/M*s2*a1';
db1 = alpha/M*sum(s1,2);
db2 = alpha/M*sum(s2);

W1 = W1-dW1;
W2 = W2-dW2;
b1 = b1-db1;
b2 = b2-db2;

%% Condition to jump out of loop
if j>1
    if J(j)>=J(j-1)
        break
    end
end


end

%% Figure plot setting
plot(p,a2,'--')
hold on
legend({'Original Function','NN approximation'})
xlabel('time')
ylabel('Output')
title('Neural Network approximation for arbitrary function')







