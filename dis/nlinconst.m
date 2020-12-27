function [c,ceq] = nlinconst(x)
% ceq rubbishes
global x1
global x2

obstacles1 = [1.23,3.47;1.75,4.00;2.10,3.63;1.58,2.30;1.40,2.67];
obstacles2 = [4.65,5.98;4.00,6.48;4.52,7.68;5.06,7.73;5.90,6.95];
obstacles3 = [6.78,3.40;7.78,5.10;7.78,3.76];
obstacles4 = [4.00,3.00;4.35,3.35;4.80,3.45;4.37,2.75];
delta_path=[x;x2]-[x1;x];
path2=[x;x2];
path1=[x1;x];

m=size(obstacles1,1);
delta=[obstacles1(2:m,:);obstacles1(1,:)]-obstacles1;
ob2=[obstacles1(2:m,:);obstacles1(1,:)];
t=((path1(:,1).*path2(:,2)-path1(:,2).*path2(:,1))*delta(:,2)'-delta_path(:,2)*(obstacles1(:,1).*ob2(:,2)-obstacles1(:,2).*ob2(:,1))')./(delta_path(:,2)*delta(:,1)'-delta_path(:,1)*delta(:,2)');
judge1=sum(double(((t-path1(:,2)).*(t-path2(:,2)))<0).*double(((t-ob2(:,2)').*(t-obstacles1(:,2)'))<0),'all');

m=size(obstacles2,1);
delta=[obstacles2(2:m,:);obstacles2(1,:)]-obstacles2;
ob2=[obstacles2(2:m,:);obstacles2(1,:)];
t=((path1(:,1).*path2(:,2)-path1(:,2).*path2(:,1))*delta(:,2)'-delta_path(:,2)*(obstacles2(:,1).*ob2(:,2)-obstacles2(:,2).*ob2(:,1))')./(delta_path(:,2)*delta(:,1)'-delta_path(:,1)*delta(:,2)');
judge2=sum(double(((t-path1(:,2)).*(t-path2(:,2)))<0).*double(((t-ob2(:,2)').*(t-obstacles2(:,2)'))<0),'all');

m=size(obstacles3,1);
delta=[obstacles3(2:m,:);obstacles3(1,:)]-obstacles3;
ob2=[obstacles3(2:m,:);obstacles3(1,:)];
t=((path1(:,1).*path2(:,2)-path1(:,2).*path2(:,1))*delta(:,2)'-delta_path(:,2)*(obstacles3(:,1).*ob2(:,2)-obstacles3(:,2).*ob2(:,1))')./(delta_path(:,2)*delta(:,1)'-delta_path(:,1)*delta(:,2)');
judge3=sum(double(((t-path1(:,2)).*(t-path2(:,2)))<0).*double(((t-ob2(:,2)').*(t-obstacles3(:,2)'))<0),'all');

m=size(obstacles4,1);
delta=[obstacles1(2:m,:);obstacles4(1,:)]-obstacles4;
ob2=[obstacles4(2:m,:);obstacles4(1,:)];
t=((path1(:,1).*path2(:,2)-path1(:,2).*path2(:,1))*delta(:,2)'-delta_path(:,2)*(obstacles4(:,1).*ob2(:,2)-obstacles4(:,2).*ob2(:,1))')./(delta_path(:,2)*delta(:,1)'-delta_path(:,1)*delta(:,2)');
judge4=sum(double(((t-path1(:,2)).*(t-path2(:,2)))<0).*double(((t-ob2(:,2)').*(t-obstacles4(:,2)'))<0),'all');


c=[judge1;judge2;judge3;judge4];
ceq=[];
end

