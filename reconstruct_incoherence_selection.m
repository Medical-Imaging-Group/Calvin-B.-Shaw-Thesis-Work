
function [fwd_mesh,pj_error] = reconstruct_incoherence_selection(fwd_mesh,...
						recon_basis,...
						data_fn,...
						iteration,...
						lambda,...
						output_fn,...
						filter_n)

%This MATLAB code is used as part of the work presented in:
%Calvin B. Shaw and Phaneendra K. Yalavarthy, “Incoherence based optimal selection of independent measurements in diﬀuse optical tomography," Journal of Biomedical Optics, (2013) (Under Review).   
%Please contact: <calvinshaw87@gmail.com> for any bugs or queries!!!                                 
% CW Reconstruction program for standard meshes
%
% fwd_mesh is the input mesh (variable or filename)
% recon_basis is the reconstruction basis (pixel basis or mesh filename)
% data_fn is the boundary data (variable or filename)
% iteration is the max number of iterations
% lambda is the initial regularization value
% output_fn is the root output filename
% filter_n is the number of mean filters




% set modulation frequency to zero.
frequency = 0;

tic;

%****************************************
% If not a workspace variable, load mesh
if ischar(fwd_mesh)== 1
  fwd_mesh = load_mesh(fwd_mesh);
end
if ~strcmp(fwd_mesh.type,'stnd')
    errordlg('Mesh type is incorrect','NIRFAST Error');
    error('Mesh type is incorrect');
end

% Load or calculate second mesh for reconstruction basis
if ischar(recon_basis)
  recon_mesh = load_mesh(recon_basis);
  [fwd_mesh.fine2coarse,...
   recon_mesh.coarse2fine] = second_mesh_basis(fwd_mesh,recon_mesh);
elseif isstruct(recon_basis)
    recon_mesh = recon_basis;
  [fwd_mesh.fine2coarse,...
   recon_mesh.coarse2fine] = second_mesh_basis(fwd_mesh,recon_mesh);
else
  [fwd_mesh.fine2coarse,recon_mesh] = pixel_basis(recon_basis,fwd_mesh);
end

% set parameters for second mesh. Jacobian calculation has been modified so
% that it only calculates Jacobian for second mesh, which is more memory
% efficient as well as computationally faster
recon_mesh.type = fwd_mesh.type;
recon_mesh.link = fwd_mesh.link;
recon_mesh.source = fwd_mesh.source;
recon_mesh.meas = fwd_mesh.meas;
recon_mesh.dimension = fwd_mesh.dimension;
if recon_mesh.dimension == 2
    recon_mesh.element_area = ele_area_c(recon_mesh.nodes(:,1:2),...
                                         recon_mesh.elements);
else
    recon_mesh.element_area = ele_area_c(recon_mesh.nodes,...
                                         recon_mesh.elements);
end                              
pixel.support = mesh_support(recon_mesh.nodes,...
                             recon_mesh.elements,...
                             recon_mesh.element_area);

% read data - This is the calibrated experimental data or simulated data
anom = load_data(data_fn);
if ~isfield(anom,'paa')
    errordlg('Data not found or not properly formatted','NIRFAST Error');
    error('Data not found or not properly formatted');
end
anom = anom.paa;
anom = log(anom(:,1));
% find NaN in data

datanum = 0;
% [ns,junk]=size(fwd_mesh.source.coord);
% for i = 1 : ns
%   for j = 1 : length(fwd_mesh.link(i,:))
%       datanum = datanum + 1;
%       if fwd_mesh.link(i,j) == 0
%           anom(datanum,:) = NaN;
%       end
%   end
% end

ind = find(isnan(anom(:,1))==1);
% set mesh linkfile not to calculate NaN pairs:
link = fwd_mesh.link';
link(ind) = 0;
fwd_mesh.link = link';
recon_mesh.link = link';
clear link
% remove NaN from data
ind = setdiff(1:size(anom,1),ind);
anom = anom(ind,:);
clear ind;

% check for input regularization
if isstruct(lambda) && ~(strcmp(lambda.type,'JJt') || strcmp(lambda.type,'JtJ'))
    lambda.type = 'Automatic';
end
if ~isstruct(lambda)
    lambda.value = lambda;
    lambda.type = 'Automatic';
end
% determine regularization type
if strcmp(lambda.type, 'Automatic')
    if size(anom,1)<size(recon_mesh.nodes,1)
        lambda.type = 'JJt';
    else
        lambda.type = 'JtJ';
    end
end

% Initiate projection error
pj_error = [];

% Initiate log file
fid_log = fopen([output_fn '.log'],'w');
fprintf(fid_log,'Forward Mesh   = %s\n',fwd_mesh.name);
if ischar(recon_basis)
  fprintf(fid_log,'Basis          = %s\n',recon_basis);
end
fprintf(fid_log,'Frequency      = %f MHz\n',frequency);
if ischar(data_fn) ~= 0
    fprintf(fid_log,'Data File      = %s\n',data_fn);
end
if isstruct(lambda)
    fprintf(fid_log,'Initial Regularization  = %d\n',lambda.value);
else
    fprintf(fid_log,'Initial Regularization  = %d\n',lambda);
end
fprintf(fid_log,'Filter         = %d\n',filter_n);
fprintf(fid_log,'Output Files   = %s_mua.sol\n',output_fn);
fprintf(fid_log,'               = %s_mus.sol **CW recon only**\n',output_fn);


for it = 1 : iteration
  
  % Calculate jacobian
  [J,data]=jacobian_stnd(fwd_mesh,frequency,recon_mesh);

  % Set jacobian as Phase and Amplitude part instead of complex
  J = J.complete;

  % Read reference data
  clear ref;
  ref = log(data.amplitude);
  
  data_diff = (anom-ref);

  pj_error = [pj_error sum(abs(data_diff.^2))]; 
  
  disp('---------------------------------');
  disp(['Iteration Number          = ' num2str(it)]);
  disp(['Projection error          = ' num2str(pj_error(end))]);

  fprintf(fid_log,'---------------------------------\n');
  fprintf(fid_log,'Iteration Number          = %d\n',it);
  fprintf(fid_log,'Projection error          = %f\n',pj_error(end));

  if it ~= 1
    p = (pj_error(end-1)-pj_error(end))*100/pj_error(end-1);
    disp(['Projection error change   = ' num2str(p) '%']);
    fprintf(fid_log,'Projection error change   = %f %%\n',p);
    if p <= 2
      disp('---------------------------------');
      disp('STOPPING CRITERIA REACHED');
      fprintf(fid_log,'---------------------------------\n');
      fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
     break
    end
  end

  % Interpolate onto recon mesh
  [recon_mesh] = interpolatef2r(fwd_mesh,recon_mesh);

  N = recon_mesh.mua;
    nn = length(recon_mesh.nodes);
    % Normalise by looping through each node, rather than creating a
    % diagonal matrix and then multiplying - more efficient for large meshes
    for i = 1 : nn
        J(:,i) = J(:,i).*N(i,1);
    end
    clear nn N

        [nrow,ncol]=size(J);
        
        

        J_org=J;
        [m n] = size(J);
     %%%%%%%%%%%%%%%Incoherence-based measurement selection starts from here%%%%%%%%%
          for ii=1:m
              ro=J(ii,:);
              J_N(ii,:)=J(ii,:)./norm(ro);%%%%%%%%make the rows of the Jacobian-unit-length rows
          end
           Dn = eye(m);
N=abs(J_N*J_N');%Coherence Matrix (C=|J*J'| as per the manuscript)

    
    th=0.7;%0.696;%0.7;%0.75;%0.696;%0.7;%0.696;%0.75;%0.7;%0.696;%0.75; %Desired threshold (th, in the manucript)
    if it == 1

         for i = 1:1:m
            
            if Dn(i,i) == 1
                x = N(:,i);
                in1 = find(x <=  th * max(x) | x == max(x));
                in2 = find(x > th * max(x) & x ~= max(x));
                
                Dn(in2, i) = 1;
                
                for iii = 1:length(in2)
                    Dn(in2(iii), in2(iii)) = 0;
                end
                Dn(i,i) = 1;
                clear in2 in1 x;
            end
            
        end
        
        in = find(diag(Dn) == 1);
% 
    end
    
%%%Note  the indices of incoherent measurements is contained in -> in

        J = J(in,:);%%Slect only the incoherent rows of Jacobian
  
    data_diff= data_diff(in);%Corresponding rows of 
    
   

%%%%%%%%%%%%%%%End of incoherence code%%%%%%%%%%

    [m n] = size(J);


% cond(J)
% ch=norm(data_diff)/(m*m)
   %%%%%%%%%%%%%%%%%%%%%
   % Add regularization
  if it ~= 1
    lambda.value = lambda.value./10^0.25;
  end
  
%  build hessian
  [nrow,ncol]=size(J)
  
  if strcmp(lambda.type, 'JJt')
      
      Hess = zeros(nrow);
      Hess = (J*J');
 
      reg_amp = lambda.value*max(diag(Hess));
      reg = ones(nrow,1);
      reg = reg.*reg_amp;

      disp(['Amp Regularization        = ' num2str(reg(1,1))]);
      fprintf(fid_log,'Amp Regularization        = %f\n',reg(1,1));

%       Add regularisation to diagonal - looped rather than creating a matrix
%       as it is computational more efficient for large meshes
      for i = 1 : nrow
          Hess(i,i) = Hess(i,i) + reg(i);
      end

%      Calculate update
      foo = J'*(Hess\data_diff);
  
  else
      
      Hess = zeros(ncol);
      Hess = (J'*J);
 
      reg_mua = lambda.value*max(diag(Hess));
      reg = ones(ncol,1);
      reg = reg.*reg_mua;

      disp(['Mua Regularization        = ' num2str(reg(1,1))]);
      fprintf(fid_log,'Mua Regularization        = %f\n',reg(1,1));

%       Add regularisation to diagonal - looped rather than creating a matrix
%       as it is computational more efficient for large meshes
      for i = 1 : ncol
          Hess(i,i) = Hess(i,i) + reg(i);
      end

      %Calculate update
      foo = Hess\J'*data_diff;
      
  end
  
   %%%%%%%%%%%%%%%%%%%%%%%%
  %%%Normalized Inner product%%%%
  %%%Normalized Inner product%%%%
 % A=J_pre;
%    B=J;
%    kk=1;
%    for ii=1:nrow
% %       
%        for jj=ii:nrow
% %           %%%for J with precondition%%%
% %           numer1=abs(sum(A(:,ii).*A(:,jj)));
% %           denom1=norm(A(:,ii),2)*norm(A(:,jj),2);
% %           rA(kk)=numer1/denom1;
% %           %%%for J without precondition%%%
%            numer2=abs(sum(B(ii,:).*B(jj,:)));
%            denom2=norm(B(ii,:),2)*norm(B(jj,:),2);
%            rB(kk)=numer2/denom2;
%            kk=kk+1;
%        end
%    end
%    inde=1:length(rB);     
% %   val1=sort(rA,'descend');
%    val2=sort(rB,'descend');
%    save rB.mat rB
%    hold on;
%    %plot(inde,val1,'*-r');
%    plot(inde,val2,'-');
% %   legend('Jpre','J')
%    hold off;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%save foo.mat foo data_diff nn J
  clear nn N
%%%%% Compressive Sensing Should end here %%%%%%%%%%%%
  
  foo =foo.*(recon_mesh.mua);
  
  % Update values
  recon_mesh.mua =recon_mesh.mua +1*foo;
  %fwd_mesh.mua = fwd_mesh.mua + (foo);

  
  
  
  jk = find(recon_mesh.mua(:) <= 0);
  recon_mesh.mua(jk) = 0.01;
  
  recon_mesh.kappa = 1./(3.*(recon_mesh.mua + recon_mesh.mus));
  %%% To see the delta_mu (update alone)
%   figure; plotimage(recon_mesh, foo);
%   
%   %%% To see the iterative mu
   %figure;
   %plotimage(recon_mesh, recon_mesh.mua);
  
  
  clear foo Hess Hess_norm tmp data_diff G

  % Interpolate optical properties to fine mesh
  [fwd_mesh,recon_mesh] = interpolatep2f(fwd_mesh,recon_mesh);
%fwd_mesh=recon_mesh;
  % We dont like -ve mua or mus! so if this happens, terminate
%   if (any(fwd_mesh.mua<0) | any(fwd_mesh.mus<0))
%     disp('---------------------------------');
%     disp('-ve mua calculated...not saving solution');
%     fprintf(fid_log,'---------------------------------\n');
%     fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
%     break
%   end
%   
  % Filtering if needed!
  if filter_n > 1
    fwd_mesh = mean_filter(fwd_mesh,abs(filter_n));
  elseif filter_n < 0
    fwd_mesh = median_filter(fwd_mesh,abs(filter_n));
  end

  if it == 1
    fid = fopen([output_fn '_mua.sol'],'w');
  else
    fid = fopen([output_fn '_mua.sol'],'a');
  end
  fprintf(fid,'solution %g ',it);
  fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
  fprintf(fid,'-components=1 ');
  fprintf(fid,'-type=nodal\n');
  fprintf(fid,'%f ',fwd_mesh.mua);
  fprintf(fid,'\n');
  fclose(fid);
  
  if it == 1
    fid = fopen([output_fn '_mus.sol'],'w');
  else
    fid = fopen([output_fn '_mus.sol'],'a');
  end
  fprintf(fid,'solution %g ',it);
  fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
  fprintf(fid,'-components=1 ');
  fprintf(fid,'-type=nodal\n');
  fprintf(fid,'%f ',fwd_mesh.mus);
  fprintf(fid,'\n');
  fclose(fid);
end

% close log file!
time = toc;
fprintf(fid_log,'Computation TimeRegularization = %f\n',time);
fclose(fid_log);





function [recon_mesh] = interpolatef2r(fwd_mesh,recon_mesh)

% This function interpolates fwd_mesh into recon_mesh
NNC = size(recon_mesh.nodes,1);

for i = 1 : NNC
  if fwd_mesh.fine2coarse(i,1) ~= 0
    recon_mesh.mua(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
    fwd_mesh.mua(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    recon_mesh.mus(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
    fwd_mesh.mus(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    recon_mesh.kappa(i,1) = (fwd_mesh.fine2coarse(i,2:end) * ...
    fwd_mesh.kappa(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
    recon_mesh.region(i,1) = ...
    median(fwd_mesh.region(fwd_mesh.elements(fwd_mesh.fine2coarse(i,1),:)));
  elseif fwd_mesh.fine2coarse(i,1) == 0
    dist = distance(fwd_mesh.nodes,...
		    fwd_mesh.bndvtx,...
		    recon_mesh.nodes(i,:));
    mindist = find(dist==min(dist));
    mindist = mindist(1);
    recon_mesh.mua(i,1) = fwd_mesh.mua(mindist);
    recon_mesh.mus(i,1) = fwd_mesh.mus(mindist);
    recon_mesh.kappa(i,1) = fwd_mesh.kappa(mindist);
    recon_mesh.region(i,1) = fwd_mesh.region(mindist);
  end
end

function [fwd_mesh,recon_mesh] = interpolatep2f(fwd_mesh,recon_mesh)

for i = 1 : length(fwd_mesh.nodes)
  fwd_mesh.mua(i,1) = ...
      (recon_mesh.coarse2fine(i,2:end) * ...
       recon_mesh.mua(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
  fwd_mesh.kappa(i,1) = ...
      (recon_mesh.coarse2fine(i,2:end) * ...
       recon_mesh.kappa(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
  fwd_mesh.mus(i,1) = ...
      (recon_mesh.coarse2fine(i,2:end) * ...
       recon_mesh.mus(recon_mesh.elements(recon_mesh.coarse2fine(i,1),:)));
end
