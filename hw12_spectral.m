function KdVrkm()
close all
fsz = 20;

init_data = 1;

N = 200;
L = N;
x = linspace(-L/2,L/2,N+1);
x(N + 1) = [];
k = -N/2 : (N/2 - 1);
u = zeros(1,N);
if init_data == 1
   u0 = cos(x/16).*(1 + sin(x/16));
end
if init_data == 2
   u0 = cos(x/16).*(1 + sin(x/16));
end
dt = 0.1;
figure; clf;
hpic = plot(x,u0,'LineWidth',2,'color','r');
hold on;
grid
xlim([-L/2,L/2]);
set(gca,'Fontsize',fsz);
xlabel('x','FontSize',fsz);
ylabel('u','FontSize',fsz);
drawnow
if init_data == 1
    hp = plot(x,u0,'LineWidth',2);
    axis([-L/2 L/2 -0.01 1.01]);
end
tmax = 120;
t = 0;
freq = k.*(2*pi/L);
freq3 = freq.^3;
e3=exp((freq.^2-freq.^4)*dt);
M = floor(tmax/dt);
uxt = zeros(N,M);
count = 0;
while (t<tmax)
    count = count + 1;
    t=t+dt;
    vhat=fftshift(fft(u0));
    k1=rhs(0,vhat);
    k2=rhs(0.5*dt,vhat+0.5*dt*k1);
    k3=rhs(0.5*dt,vhat+0.5*dt*k2);
    k4=rhs(dt,vhat+dt*k3);
    vhat_new=vhat+dt*(k1+2*k2+2*k3+k4)/6;
    unew=ifft(ifftshift(e3.*vhat_new));
    set(hpic,'xdata',x,'ydata',real(unew));
    if init_data == 1
        y = -N/2 + mod(x - t/3 + N/2,N);
        set(hp,'xdata',x,'ydata',1./(cosh((y)/sqrt(12))).^2);
        axis([-N/2 N/2 -0.01 1.01]);
    end
    u0=unew;
    drawnow;
    uxt(:,count)=u0;
end
figure;
imagesc(x(N/2:N),(0:dt:tmax),real(uxt)');
xlabel('x');
ylabel('t');
title('u(x,t)');
end

function RHSvhat=rhs(dt,vhat)
N=size(vhat,2);
L=N;
k=-N/2:(N/2-1);
freq=k.*(2*pi/L);
freq3=freq.^3;
e3=exp(1i*freq3*dt);
em3=exp(-1i*freq3*dt);
vhat1=vhat.*e3;
v1=ifft(ifftshift(vhat1));
v2=0.5*v1.^2;
RHSvhat=-em3.*(1i*freq).*fftshift(fft(v2));
end