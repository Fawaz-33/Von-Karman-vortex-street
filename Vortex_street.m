clear all
close all
clc
%% Grid Parameters

nx = 200;
ny = 200;
[X,Y] = meshgrid(1:nx,1:ny);

ux(nx,ny) = 0; uy(nx,ny) = 0; density(nx,ny) = 0; p(nx,ny) = 0 ; % Initial values of velocity,pressure, density

density0 = 0.05;


%% MAIN SOLVER

f = cell(2500,1);
for iter = 1:2500 
    
    % Take divergence of velocity
    divg = divergence(ux,uy);
    
    % Solve Pressure Poisson equation with function
    p = laplacian(divg);
    
    % Apply pressure boundary condition
    p(1:nx,1)=p(1:nx,2);
    p(1,1:nx)=p(2,1:nx);
    p(nx,1:ny)=p(nx-1,1:ny);
    p(1:nx,ny)=p(1:nx,ny-1);
    
    % Take gradient of pressure and subtract from velocity field
    % to get divergence free velocity
    
    [gpx,gpy]=gradient(p);
    ux = ux - gpx;
    uy = uy - gpy;
    
    % Using 4th Order RK method for advection (Backstep)
    [ux_new,uy_new] = RK4(X,Y,ux,uy,-1);
    ux = interp2(X,Y,ux,ux_new,uy_new);
    uy = interp2(X,Y,uy,ux_new,uy_new);
    density = interp2(X,Y,density,ux_new,uy_new);
    
    % Apply velocity boundary condition 
    
    ux(1,1:ny)=0;
    ux(nx,1:ny)=0;
    ux(1:nx,ny)=0;
    ux(1:nx,1)=0;
    uy(1,1:ny)=0;
    uy(nx,1:ny)=0;
    uy(1:nx,ny)=0;
    uy(1:nx,1)=0;
    
    % Apply inlet 
    ux(1:200,5:17) = 2.5;
    ux(97:103,30:34) = 0; % Square obstruction
    density(1:200,5:17) = density0;
    density(97:103,30:34) = 0; % Square obstruction
    pcolor(X,Y,density)
    colormap('hot')
    shading interp 
    
    drawnow
    
    f{iter}=getframe(gcf);
    
    
end
 % For video
 obj = VideoWriter('JoStamVS.avi');
 obj.Quality = 100;
 obj.FrameRate =3;
 open(obj);
 for iter = 1:2500
     writeVideo(obj,f{iter});
 end
obj.close();

%% Laplacian operator function

function Xx=laplacian(div)
[nx,ny]=size(div);
Boundary_index = [   1:nx  ,   1:nx:1+(ny-1)*nx ,  1+(ny-1)*nx:nx*ny,   nx:nx:nx*ny ]; % Boundary Locations
diagonals = [-4*ones(nx*ny,1),ones(nx*ny,4)]; % For diagonals
A = spdiags(diagonals,[0 -1 1 -nx nx],nx*ny,nx*ny); % Laplacian operator
I=speye(nx*ny);
A(Boundary_index,:)=I(Boundary_index,:); %Changing boundary values of A matrix
b = zeros(nx,ny);
b = div;
b = reshape(b,nx*ny,1);
Xx = A\b;
Xx = reshape(Xx,nx,ny);
end



%% 4th Order Runge-Kutta Function

function [x_new,y_new] = RK4(px,py,vx,vy,h)
   k1x = interp2(vx,px,py);
   k1y = interp2(vy,px,py);
   k2x = interp2(vx,px+h/2*k1x,py+h/2*k1y);
   k2y = interp2(vy,px+h/2*k1x,py+h/2*k1y);
   k3x = interp2(vx,px+h/2*k2x,py+h/2*k2y);
   k3y = interp2(vy,px+h/2*k2x,py+h/2*k2y);
   k4x = interp2(vx,px+h*k3x,py+h*k3y);
   k4y = interp2(vy,px+h*k3x,py+h*k3y);
   x_new = (px+h*(k1x+2*k2x+2*k3x+k4x)/6);
   y_new = (py+h*(k1y+2*k2y+2*k3y+k4y)/6);
end