clear all;
% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]
% create the time array
time.dt = 5e-8;         % sampling time in sec 5e-8
time.length = 500;      % number of points in time

object_sim.Nx = 501;  % number of grid points in the x (row) direction
object_sim.Ny = 501;  % number of grid points in the y (column) direction
object_sim.x = 50.1e-3;              % total grid size [m]
object_sim.y = 50.1e-3;              % total grid size [m]

dx = object_sim.x/object_sim.Nx;              % grid point spacing in the x direction [m]
dy = object_sim.y/object_sim.Ny;              % grid point spacing in the y direction [m]
kgrid = makeGrid(object_sim.Nx, dx, object_sim.Ny, dy);

% create a second computation grid for the reconstruction to avoid the
% inverse crime
object_rec.Nx = 501;  % number of grid points in the x (row) direction
object_rec.Ny = 501;  % number of grid points in the y (column) direction
object_rec.x = 50.1e-3;              % total grid size [m]
object_rec.y = 50.1e-3;              % total grid size [m]

dx_rec = object_rec.x/object_rec.Nx;              % grid point spacing in the x direction [m]
dy_rec = object_rec.y/object_rec.Ny;              % grid point spacing in the y direction [m]
recon_grid = makeGrid(object_rec.Nx, dx_rec, object_rec.Ny, dy_rec);

% define a centered Cartesian circular sensor
clear sensor;
sensor_radius = 22e-3;     % [m]
sensor_angle = 2*pi;      % [rad]
sensor_pos = [0, 0];        % [m]
num_sensor_points = 60;
cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);
sensor.mask = cart_sensor_mask;
%    sensor.frequency_response = [2.25e6 70];

M = 100;
N = 100;
indxi = ceil(object_sim.Nx/2) - M:ceil(object_sim.Nx/2) + M;
indyi = ceil(object_sim.Ny/2) - N:ceil(object_sim.Ny/2) + N;

Nxi = length(indxi);
Nyi = length(indyi);
tl = time.length;
sml = length(sensor.mask);
ANx = tl*sml;
ANy = Nxi*Nyi;

tic
[A_b] = sysBuildPAT_mod_Band(object_sim,indxi,indyi,time,medium,sensor,sensor_radius);
toc


signal_to_noise_ratio = 20;	% [dB]
scaling_factor = 0.15;
IM_init_v = zeros((2*M+1)*(2*N+1),1);

%%%%%%%%%%%%%%%%%%%% Derenzo Phantom - Forward Data %%%%%%%%%%%%%%%%%%
bv = imread('derenzo.png');
bv = double(rgb2gray(bv));
ind = find(bv(:)<120);
bv(ind) = 0;
ind = find(bv(:)>120);
bv(ind)=1;
BV1 = imresize(double(bv),[230 230]);
ind = find(BV1(:)<0.9);
BV1(ind) = 0;
ind = find(BV1(:)>0.9);
BV1(ind) = 1;

BV2 = BV1(10:210,20:220);
object_sim.p0 = zeros(object_sim.Nx, object_sim.Ny);
object_sim.p0(indxi,indyi) = BV2(:,:);
sd2 = forward(object_sim, time, medium, sensor);
% add noise to the recorded sensor data
sdn2 = addNoise(sd2, signal_to_noise_ratio, 'peak');
sdn2_v = reshape(sdn2,ANx,1);



%%%%%%%%%%%%%%%%%%%% PAT Phantom - Forward Data %%%%%%%%%%%%%%%%%%
bv = imread('PAT_1.jpg');
bv = double(rgb2gray(bv));
bv = medfilt2(bv,[3 3]);
ind = find(bv(:)<150);
bv(ind) = 0;
ind = find(bv(:)>150);
bv(ind)=1;
BV1 = imresize(double(bv),[210 210],'bicubic');
ind = find(BV1(:)<0.7);
BV1(ind) = 0;
ind = find(BV1(:)>0.7);
BV1(ind) = 1;
BV2 = (BV1(5:205,5:205));
BV2 = medfilt2(BV2,[3 3]);
SE = strel('square',2)
BV2 = imdilate(BV2,SE);
BV3 = edge(BV2,'sobel');
ind = find(BV3(:)>0.5);
BV2(ind)=1;
ind = find(BV2(:)==1);
BV2(ind) = 0.5;
ind = find(BV2(:)==0);
BV2(ind) = 1;
ind = find(BV2(:)==0.5);
BV2(ind) = 0;
figure; imshow(BV2,[]);
% SE = strel('square',6)
% BV2 = imdilate(BV2,SE);
object_sim.p0 = zeros(object_sim.Nx, object_sim.Ny);
object_sim.p0(indxi,indyi) = BV2(:,:);
sd2 = forward(object_sim, time, medium, sensor);
% add noise to the recorded sensor data
sdn2 = addNoise(sd2, signal_to_noise_ratio, 'peak');
sdn2_v = reshape(sdn2,ANx,1);



%%%%%%%%%%%% K-Wave reconstructions %%%%%%%%%%%%%%%%%%

% create a binary sensor mask of an equivalent continuous circle
sensor_radius_grid_points = round(sensor_radius/dx_rec);
binary_sensor_mask = makeCircle(object_rec.Nx, object_rec.Ny, object_rec.Nx/2, object_rec.Ny/2, sensor_radius_grid_points, sensor_angle);
sensor.mask = binary_sensor_mask;
sdn2_interp = interpCartData(recon_grid, sdn2, cart_sensor_mask, binary_sensor_mask);
IM2_k_interp = inverse(object_rec, time, medium, sensor, sdn2_interp);

%%%%%%%%%%%% Model Based Reconstruction %%%%%%%%%%%%%%%
% Backprojection Based
IM2_v = A'*(scaling_factor*sdn2_v);
IM2_bp = reshape(IM2_v,2*M+1,2*N+1);



%%%%%%%%%%%%%%Code for Model Resolution based BPD using LSQR based image reconstruction using
%%%%%%%%%%%%%% 0.01 as and k=25 regularization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic;
l_step =25;


[U1,V1,B1] =lsqr_b_hybrid(A_b*0.15,sdn2_v,l_step,1);
T = eye(l_step);
lambda= 0.01;

reg=lambda;
T = eye(l_step);
Hess = B1(1:l_step,1:l_step+1)'*B1(1:l_step,1:l_step+1);
yk = (Hess + reg.*eye(size(B1(1:l_step,1:l_step+1),2)))\(norm(sdn2_v,2).*B1(1:l_step,1:l_step+1)'*T(:,1));
K = B1'*B1;
N3 = (K + (lambda.*eye(size((K),2))))\(K);

lambda1 = 0.00001;                                      % lambda : regularization parameter
Nit = 5000;                                           % Nit : number of iterations
mu1 = 0.05;                                           % mu : ADMM parameter

% Run BPD algorithm
[x_BPD, cost] = bpd_salsa_sparsemtx(yk, N3, lambda1, mu1, Nit);
foo = V1(:,1:length(yk))*x_BPD;

IM1_LSQR = reshape(foo,2*100+1,2*100+1);
toc;

%%%%%%%% PC for the above method
COV =  cov(BV2,IM1_LSQR);
Nr =  COV(2,1) ;
Dr =sqrt(COV(1,1) *COV(2,2))  ;
PC = Nr/Dr

%%%%%%%%%%%%%%Code for Model Resolution based BPD using LSQR based image reconstruction using
%%%%%%%%%%%%%% Optimal Choice of Regularization as and k=25 regularization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  for i=2:50
tic;
l_step =25;


[U1,V1,B1] =lsqr_b_hybrid(A_b*0.15,sdn2_v,l_step,1);
T = eye(l_step);
lambda=1000;
lambda = fminbnd(@(lambda) opt_lambda_cw(B1(1:l_step,1:l_step+1),sdn2_v, lambda, V1, A_b*0.15, T, U1), 0, lambda, optimset( 'MaxIter',1000, 'TolX', 1e-16));

reg = lambda;
T = eye(l_step);
Hess = B1(1:l_step,1:l_step+1)'*B1(1:l_step,1:l_step+1);
yk = (Hess + reg.*eye(size(B1(1:l_step,1:l_step+1),2)))\(norm(sdn2_v,2).*B1(1:l_step,1:l_step+1)'*T(:,1));

K = B1'*B1;
N3 = (K + (lambda.*eye(size((K),2))))\(K);


lambda1 = 0.00001;                                      % lambda : regularization parameter
Nit = 10000;                                           % Nit : number of iterations
mu1 = 0.01;                                           % mu : ADMM parameter

% Run BPD algorithm
[x_BPD, cost] = bpd_salsa_sparsemtx(yk, N3, lambda1, mu1, Nit);


foo = V1(:,1:length(yk))*x_BPD;


IM4_LSQR = reshape(foo,2*100+1,2*100+1);
toc;
%%%%%%%% PC for the above method
COV =  cov(BV2,IM4_LSQR);
Nr =  COV(2,1) ;
Dr =sqrt(COV(1,1) *COV(2,2))  ;
PC = Nr/Dr


%%%%%%%%%%%%%%Code for L1-norm based image reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic;
IM2_L10 = LpSynthesis(A_b, A_b', sdn2_v,1,1, 1e-6,230,1e-4,0.6);%LP Algorithm
toc
%%%%%%%% PC for the above method
COV =  cov(BV2,IM2_L10);
Nr =  COV(2,1) ;
Dr =sqrt(COV(1,1) *COV(2,2))  ;
PC = Nr/Dr

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

