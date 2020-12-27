clear;
x_all=[0.285,0.35;0.4,3;2.9,2.2;1.1,6.5;2.9,5.0;3,8.3;5.2,4.8;7.4,7.4;
    5.3,1.2;7.8,2.6;6,6;9,4.8;5,8.5;7,1.5;2.5,7.5];
global x1
global x2
num=3;

dis=zeros(15,15);
fun1=@(x) cost(x);
fun2=@(x) nlinconst(x);
for i=1:15
    for j=(i+1):15
        x1=x_all(i,:);
        x2=x_all(j,:);
        problem = createOptimProblem('fmincon','x0',ones(num,2),'objective',fun1,'nonlcon',fun2,'lb',0*ones(num,2),'ub',10*ones(num,2));
        ms = MultiStart('FunctionTolerance',2e-4,'UseParallel',true);
        gs=GlobalSearch(ms);
        dis(i,j)=cost(run(gs,problem));
    end
end
dis=dis+dis';

save('distance.mat','dis');
