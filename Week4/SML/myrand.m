function x=myrand(n)

x=zeros(1,n); u=rand(1,n);

flag=(0<=u & u<1/8);
x(flag)=sqrt(8*u(flag));
flag=(1/8<=u & u<1/4);
x(flag)=2-sqrt(2-8*u(flag));
flag=(1/4<=u & u<1/2);
x(flag)=1+4*u(flag);
flag=(1/2<=u & u<3/4);
x(flag)=3+sqrt(4*u(flag)-2);
flag=(3/4<=u & u<=1);
x(flag)=5-sqrt(4-4*u(flag));
