function length = cost(x)
global x1
global x2

delta=[x;x2]-[x1;x];
length=sum(sqrt(sum(delta.^2,2)));
end

