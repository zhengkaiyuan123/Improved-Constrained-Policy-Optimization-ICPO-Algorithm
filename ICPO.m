%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improved Constrained Policy Optimization (ICPO) Algorithm
%
% Description:
% This MATLAB implementation of the ICPO algorithm enhances the original 
% Competitive Population Optimization (CPO) by improving exploration, 
% exploitation, and adaptive population size control for solving complex 
% optimization problems.
%
% Usage:
% [Gb_Fit, Gb_Sol, Conv_curve] = ICPO(Pop_size, Tmax, lb, ub, dim, fobj);
%
% Inputs:
% - Pop_size: Population size
% - Tmax: Maximum iterations
% - lb, ub: Lower and upper bounds (vectors)
% - dim: Problem dimensionality
% - fobj: Objective function to minimize
%
% Outputs:
% - Gb_Fit: Best fitness value
% - Gb_Sol: Best solution found
% - Conv_curve: Convergence curve
%
% License:
% Open-source under the MIT License. Use, modify, and share with attribution.
%
% Author: Kaiyuan Zheng
% GitHub: https://github.com/[YourGitHubUsername]/ICPO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gb_Fit,Gb_Sol,Conv_curve]=ICPO(Pop_size,Tmax,lb,ub,dim,fobj)
Conv_curve=zeros(1,Tmax);
ub=ub.*ones(1,dim);
lb=lb.*ones(1,dim);
N=Pop_size;
N_min=round(0.8*Pop_size);
T=2;
alpha=0.2;
Tf=0.8;
X=initialization(Pop_size,dim,ub,lb);
t=0;
for i=1:Pop_size
    fitness(i)=fobj(X(i,:));
end
[Gb_Fit,index]=min(fitness);
Gb_Sol=X(index,:);    
Xp=X;

while t<=Tmax 
    r2=rand;
    for i=1:Pop_size
        
        U1=rand(1,dim)>rand;
        if rand<rand
            if rand<rand
                y1=(X(i,:)+X(randi(Pop_size),:))/2;
                X(i,:)=X(i,:)+(randn).*abs(2*rand*Gb_Sol-y1);
            else
                y2=(Gb_Sol+X(i,:)+X(randi(Pop_size),:))/3;
                X(i,:)=(U1).*X(i,:)+(1-U1).*(y2+rand*(X(randi(Pop_size),:)-X(randi(Pop_size),:)));
            end
        else
             Yt=2*rand*(1-t/(Tmax))^(t/(Tmax));
             U2=rand(1,dim)<0.5*2-1;
             S=rand*U2;
            if rand<Tf
                St=exp(fitness(i)/(sum(fitness)+eps));
                S=S.*Yt.*St;
                X(i,:)= (1-U1).*X(i,:)+U1.*(X(randi(Pop_size),:)+St*(X(randi(Pop_size),:)-X(randi(Pop_size),:))-S); 
            else
                Mt=exp(fitness(i)/(sum(fitness)+eps));
                vt=X(i,:);
                Vtp=X(randi(Pop_size),:);
                RT= (1-t/Tmax)*tan(pi/2*rand(1,1));
                Ft=rand(1,dim).*(Mt*(-vt+Vtp));              
                S=S.*Yt.*Ft;
                X(i,:)= (Gb_Sol+RT.*(round(r2+1).*Gb_Sol-X(i,:)))-S;
            end
        end
        for j=1:size(X,2)
            if  X(i,j)>ub(j)
                X(i,j)=lb(j)+rand*(ub(j)-lb(j));
            elseif  X(i,j)<lb(j)
                X(i,j)=lb(j)+rand*(ub(j)-lb(j));
            end

        end  
         nF=fobj(X(i,:));
        if  fitness(i)<nF
            X(i,:)=Xp(i,:);    
        else
            Xp(i,:)=X(i,:);
            fitness(i)=nF;
            if  fitness(i)<=Gb_Fit
                Gb_Sol=X(i,:);
                Gb_Fit=fitness(i);
            end
        end

    end
    pop = Pop_size;
    x = X;
    K = randi(pop,1);
    [~,index]=sort(fitness);
    temp=zeros(1,dim);
    for i=1:K
        temp=temp+x(index(i),:);
    end
    MM=temp./K;
    TempPos=zeros(pop,dim);
    for j=1:pop
        TempPos(j,:)=2.*MM-x(j,:); 
    end

    for i = 1:Pop_size
        if fobj(TempPos(i,:))<fobj(X(i,:))
            X(i,:) = max(min(TempPos(i,:),ub),lb);
        end
        if fobj(X(i,:))<Gb_Fit
            Gb_Fit = fobj(X(i,:));
            Gb_Sol = X(i,:); 
        end
    end

    Pop_size=fix(N_min+(N-N_min)*(1-(rem(t,Tmax/T)/Tmax/T)));
    t=t+1;
    if t>Tmax
        break
    end
    Conv_curve(t)=Gb_Fit;
 end
end
function Positions=initialization(SearchAgents_no,dim,ub,lb)
Boundary_no= size(ub,2);
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=Good_point_set(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
function [Good_points] = Good_point_set(m,n)
g = [1:m]'*ones(1,n);
Ind = [1:n];
prime1 = primes(100*n);
[~,q]=find(prime1>=(2*n+3));
p=prime1(1,q(1));
tmp2 = 2*cos((2*pi.*Ind)/p);
h = ones(m,1)*tmp2;
G_p = g.*h;
Good_points = mod(G_p,1);
end
