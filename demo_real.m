% =========================================================================
% This code compares different specral unmixing algorithms that consider
% spectral variability using realistic synthetically generated endmember 
% data.
% 
% The code is provided as part of the following publication:
%     Spectral Variability in Hyperspectral Data Unmixing: A 
%     Comprehensive Review
% 
% =========================================================================

clear all
close all
clc

warning off

pkg load statistics signal image

addpath(genpath(pwd))
%addpath(genpath('synthetic_data_generation'))
%addpath(genpath('library_extraction'))
%addpath(genpath('utils'))
%addpath('methods')

% rng(1000,'twister'); % for reproducibility, if desired

% select the signal to noise ratio:
%SNR = 30;

% generate synthetic data (abundance, endmember signatures, mixed image)
%[Y,Yim,A,A_cube,Mth,M_avg]=generate_image(SNR);

% Y - mixed image ordered as a matrix (bands * pixels)
% Yim - mixed image ordered as a tensor (rows * cols * bands)
%IM_DIR = '/home/marko/projekti/esa/esa/data/demo_images/007';
IM_DIR = '/home/marko/projekti/esa/esa';
imgs_glob = 'IMG_1405*';
% NOTE: take care of order of spectral channels; needs to be BGR[RE][IR]
imgs_paths = glob([IM_DIR '/' imgs_glob])
imgs = {};

resize_scale = 0.25;  % images are large, so reduce computational complexity
for ind = 1:length(imgs_paths)
  imgs{ind} = imresize(imread(imgs_paths{ind}), resize_scale);
end

[nr_rows, nr_cols] = size(imgs{1});
nr_pix = numel(imgs{1});
nr_bands = numel(imgs);
Y = zeros(nr_bands, nr_pix);
Yim = zeros(nr_rows, nr_cols, nr_bands);
for ind = 1:numel(imgs)
  Y(ind, :) = reshape(imgs{ind}, 1, []);
  Yim(:, :, ind) = imgs{ind};
end

% NORMALIZATION
max_val = max(max(Y));
Y = Y / max_val;
Yim = Yim / max_val;

% vegetation, soil, water, road
% load reference endmembers
ref_endmem = load('endmembers_b.mat');  % loads matrix M of reference endmembers (198x4), and reference names in cood
M = ref_endmem.M;
cood = ref_endmem.cood;
Slect_bands = load('EM_b_SlectBands.mat');  % loads selected bands from AVIRIS_wavelengths (198x1)
wl = load('AVIRIS_wavelengths.mat');  % loads 224x1 vector af AVIRIS_wavelengths
wl = wl.AVIRIS_wavlen;
% selected bands (in endmembers_b)
wl = wl(Slect_bands.SlectBands);

% Relevent wavelengths for MicaSense RedEdge MX camera [B, G, R, RE, NI]
MicaSense_wavelens = [475, 560, 668, 717, 842];
aviris_inds = [];
for ind = 1:length(MicaSense_wavelens)
  [~, I] = min(abs(wl - MicaSense_wavelens(ind)));
  aviris_inds(ind) = I;
end

M_vegetation = M(:,1);
M_water      = M(:,2);
M_dirt       = M(:,3);
M_road       = M(:,4);

% Define relevant indexes from AVIRIS_wavelengths, i.e. Slect_bands subset
M_avg = M(aviris_inds, :)

P = size(M_avg,2);  % number of spectral profiles (distinct materials) present in the scene needs to be estimated or assumed
[nr,nc,L] = size(Yim);

% extract reference endmember and library from the image
M0 = vca(Y,'Endmembers',P)
% This assumed knowledge of true endmembers, but it is only for easier comparison
M0 = sort_endmembers_to_ref(M_avg,M0);

% spectral library extraction
bundle_nbr = 5; % number of VCA runs
percent = 20; % percentage of pixels considered in each run
Lib = extractbundles_batchVCA(Y, M0, bundle_nbr, percent)

% uncomment to plot the extracted spectral library:
%{
figure
subplot(1,3,1)
plot(linspace(0.4,2.45,198), Lib{1})
xlim([0.4 2.45]), ylim([0 0.5])
xlabel('Wavelength [$\mu$m]','interpreter','latex','fontsize',14)
ylabel('Reflectance','interpreter','latex','fontsize',14)
subplot(1,3,2)
plot(linspace(0.4,2.45,198), Lib{2})
xlim([0.4 2.45]), ylim([0 0.9])
xlabel('Wavelength [$\mu$m]','interpreter','latex','fontsize',14)
subplot(1,3,3)
plot(linspace(0.4,2.45,198), Lib{3})
xlim([0.4 2.45]), ylim([0 0.15])
xlabel('Wavelength [$\mu$m]','interpreter','latex','fontsize',14)
%}



%%
% *********************************************************************** %
% perform unmixing with the different algorithms
% *********************************************************************** %

% ----------------------------------------
% FCLS (baseline)
fcls = true
if fcls
  st_time = time();
  'FCLS (baseline)'
  [A_FCLS,M_FCLS,time_fcls,Yhat_FCLS] = adaptor_FCLS(Yim,M0,Lib,[]);
  % saves the performance metrics:
  algnames{1} = 'FCLS';
  RMSEA(1) = nan;  % norm((A-A_FCLS)/numel(A),'fro');
  RMSEY(1) = norm((Y-Yhat_FCLS)/numel(Y),'fro');
  RMSEM(1) = nan;
  SAMM(1)  = nan;
  TIMES(1) = time_fcls;
  A_est{1} = A_FCLS;
  M_est{1} = nan;
  FCLS_time = time() - st_time
end


% ----------------------------------------
% Set the initialization of some of the other algorithms:
A_init = A_FCLS;



% ----------------------------------------
% MESMA
mesma = false
if mesma
  st_time = time();
  'MESMA'
  [A_MESMA,M_MESMA,time_MESMA,Yhat_MESMA] = adaptor_MESMA(Yim,M0,Lib,A_init);
  % saves the performance metrics:
  algnames{end+1} = 'MESMA';
  RMSEA(end+1) = nan;  % norm((A-A_MESMA)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_MESMA)/numel(Y),'fro');
  RMSEM(end+1) = nan;  % norm((Mth(:)-M_MESMA(:))/numel(Mth),'fro');
  %temp = 0; for i=1:nr*nc, for j=1:P, temp = temp + subspace(Mth(:,j,i),M_MESMA(:,j,i))/(nr*nc); end, end; SAMM(end+1) = temp; 
  TIMES(end+1) = time_MESMA;
  A_est{end+1} = A_MESMA;
  M_est{end+1} = M_MESMA;
  MESMA_time = time() - st_time
end


% ----------------------------------------
% Fractional Sparse SU
frac_sparse = true
if frac_sparse
  st_time = time();
  'Fractional Sparse SU'
  opt_SocialSparseU.fraction = 1/10;
  opt_SocialSparseU.lambda = 0.1;
  [A_SocialSparseU,M_SocialSparseU,time_socialSparseU,Yhat_SocialSparseU] = adaptor_SocialSparseU(Yim,M0,Lib,A_init,opt_SocialSparseU);
  % saves the performance metrics:
  algnames{end+1} = 'Fractional';
  RMSEA(end+1) = nan;  % norm((A-A_SocialSparseU)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_SocialSparseU)/numel(Y),'fro');
  RMSEM(end+1) = nan;  % norm((Mth(:)-M_SocialSparseU(:))/numel(Mth),'fro');
  %temp = 0; for i=1:nr*nc, for j=1:P, temp = temp + subspace(Mth(:,j,i),M_SocialSparseU(:,j,i))/(nr*nc); end, end; SAMM(end+1) = temp; 
  TIMES(end+1) = time_socialSparseU;
  A_est{end+1} = A_SocialSparseU;
  M_est{end+1} = M_SocialSparseU;
  frac_sparse_time = time() - st_time
end



% ----------------------------------------
% ELMM
elmm = false  % slow for large-resolution images
if elmm
  'ELMM'
  st_time = time();
  opt_elmm.lambda_s = 1;
  opt_elmm.lambda_a = 0.05;
  opt_elmm.lambda_psi = 0.01;
  [A_ELMM,M_ELMM,time_elmm,Yhat_ELMM] = adaptor_ELMM(Yim,M0,Lib,A_init,opt_elmm);
  % saves the performance metrics:
  algnames{end+1} = 'ELMM';
  RMSEA(end+1) = nan;  % norm((A-A_ELMM)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_ELMM)/numel(Y),'fro');
  RMSEM(end+1) = nan;  % norm((Mth(:)-M_ELMM(:))/numel(Mth),'fro');
  %temp = 0; for i=1:nr*nc, for j=1:P, temp = temp + subspace(Mth(:,j,i),M_ELMM(:,j,i))/(nr*nc); end, end; SAMM(end+1) = temp; 
  TIMES(end+1) = time_elmm;
  A_est{end+1} = A_ELMM;
  M_est{end+1} = M_ELMM;
  ELMM_time = time() - st_time
end



% ----------------------------------------
% DeepGUn
deepgun = false  % problems with calling python from octave
if deepgun
  'DeepGUn'
  opt_DeepGUn.dimAut = 2;
  opt_DeepGUn.lambda_zref = 0.001;
  opt_DeepGUn.lambda_a = 0.01;
  [A_DeepGUn,M_DeepGUn,time_DeepGUn,Yhat_DeepGUn] = adaptor_DeepGUn(Yim,M0,Lib,A_init,opt_DeepGUn);
  % saves the performance metrics:
  algnames{end+1} = 'DeepGUn';
  RMSEA(end+1) = nan;  % norm((A-A_DeepGUn)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_DeepGUn)/numel(Y),'fro');
  RMSEM(end+1) = nan;  % norm((Mth(:)-M_DeepGUn(:))/numel(Mth),'fro');
  %temp = 0; for i=1:nr*nc, for j=1:P, temp = temp + subspace(Mth(:,j,i),M_DeepGUn(:,j,i))/(nr*nc); end, end; SAMM(end+1) = temp; 
  TIMES(end+1) = time_DeepGUn;
  A_est{end+1} = A_DeepGUn;
  M_est{end+1} = M_DeepGUn;
end



% ----------------------------------------
% RUSAL
rusal = false
if rusal
  st_time = time();
  'RUSAL'
  opt_RUSAL.tau = 0.001;
  opt_RUSAL.tau2 = 0.001;
  [A_RUSAL,M_RUSAL,time_RUSAL,Yhat_RUSAL] = adaptor_RUSAL(Yim,M0,Lib,A_init,opt_RUSAL);
  % saves the performance metrics:
  algnames{end+1} = 'RUSAL';
  RMSEA(end+1) = nan;  % norm((A-A_RUSAL)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_RUSAL)/numel(Y),'fro');
  RMSEM(end+1) = nan;
  SAMM(end+1)  = nan; 
  TIMES(end+1) = time_RUSAL;
  A_est{end+1} = A_RUSAL;
  M_est{end+1} = nan;
  RUSAL_time = time() - st_time
end


% ----------------------------------------
% Normal Compositional Model
ncm = false
if ncm
  'Normal Compositional Model'
  opt_NCM = [];
  [A_NCM,M_NCM,time_NCM,Yhat_NCM] = adaptor_NCM(Yim,M0,Lib,A_init,opt_NCM);
  % saves the performance metrics:
  algnames{end+1} = 'NCM';
  RMSEA(end+1) = nan;  % norm((A-A_NCM)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_NCM)/numel(Y),'fro');
  RMSEM(end+1) = nan;
  SAMM(end+1)  = nan; 
  TIMES(end+1) = time_NCM;
  A_est{end+1} = A_NCM;
  M_est{end+1} = nan;
end



% ----------------------------------------
% Beta Compositional Model
bcm = false
if bcm
  'Beta Compositional Model'
  [A_BCM,M_BCM,time_BCM,Yhat_BCM] = adaptor_BCM(Yim,M0,Lib,A_init);
  % saves the performance metrics:
  algnames{end+1} = 'BCM';
  RMSEA(end+1) = nan;  % norm((A-A_BCM)/numel(A),'fro');
  RMSEY(end+1) = norm((Y-Yhat_BCM)/numel(Y),'fro');
  RMSEM(end+1) = nan;
  SAMM(end+1)  = nan; 
  TIMES(end+1) = time_BCM;
  A_est{end+1} = A_BCM;
  M_est{end+1} = nan;
end



%%
% *********************************************************************** %
% show quantitative results
% *********************************************************************** %
scaleV = 10000; % scale the numbers to make vizualization easier

pad_string = '...............';

interpreter = 'tex';

fprintf('\n\n Abundance estimation results: \n')
for i=1:length(algnames)
    %fprintf([pad(algnames{i},15,'right','.') ': %f \n'], scaleV*RMSEA(i))
    fprintf([algnames{i} pad_string ': %f \n'], scaleV*RMSEA(i))
end

fprintf('\n\n RMSY results: \n')
for i=1:length(algnames)
    %fprintf([pad(algnames{i},15,'right','.') ': %f \n'], scaleV*RMSEY(i))
    fprintf([algnames{i} pad_string ': %f \n'], scaleV*RMSEY(i))
end

fprintf('\n\n RMSM results: \n')
for i=1:length(algnames)
    %fprintf([pad(algnames{i},15,'right','.') ': %f \n'], scaleV*RMSEM(i))
    fprintf([algnames{i} pad_string ': %f \n'], scaleV*RMSEM(i))
end

%fprintf('\n\n SAMM results: \n')
%for i=1:length(algnames)
%    %fprintf([pad(algnames{i},15,'right','.') ': %f \n'], scaleV*SAMM(i))
%    fprintf([algnames{i} pad_string ': %f \n'], scaleV*SAMM(i))
%end

fprintf('\n\n Times results: \n')
for i=1:length(algnames)
    %fprintf([pad(algnames{i},15,'right','.') ': %f \n'], TIMES(i))
    fprintf([algnames{i} pad_string ': %f \n'], TIMES(i))
end




%%
% *********************************************************************** %
% show visual results
% *********************************************************************** %
FSize = 14; % fontsize for the labels

% plot the abundances ---------------------------------
fh = figure;
%[ha, pos] = tight_subplot(length(algnames)+1, 3, 0.01, 0.1, 0.1);
[ha, pos] = tight_subplot(length(algnames), P, 0.01, 0.1, 0.1);

%axes(ha(1));
%imagesc(A_cube(:,:,1),[0 1]), set(gca,'ytick',[],'xtick',[])%, axis square
%axes(ha(2));
%imagesc(A_cube(:,:,2),[0 1]), set(gca,'ytick',[],'xtick',[])%, axis square
%axes(ha(3));
%imagesc(A_cube(:,:,3),[0 1]), set(gca,'ytick',[],'xtick',[])%, axis square

for i=1:length(algnames)
    for j=1:P
      axes(ha(j + (i-1)*P));
      imagesc(reshape(A_est{i}(j, :), [nr nc]), [0 1]), set(gca,'ytick',[],'xtick',[])  %, axis square
      %imagesc(reshape(A_est{i}(1,:),[nr nc]),[0 1]), set(gca,'ytick',[],'xtick',[])%, axis square
      %axes(ha(2 + (i-0)*3));
      %axes(ha(i+1))
      %imagesc(reshape(A_est{i}(2,:),[nr nc]),[0 1]), set(gca,'ytick',[],'xtick',[])%, axis square
      %axes(ha(3 + (i-0)*3));
      %imagesc(reshape(A_est{i}(3,:),[nr nc]),[0 1]), set(gca,'ytick',[],'xtick',[])%, axis square
    end
    
    %axes(ha(1 + (i-0)*3));
    axes(ha(1 + (i-1) * P));
    ylabel(algnames{i},'interpreter',interpreter,'FontSize',FSize)
end

%material_names = []

axes(ha(1));
ylabel('Reference','interpreter',interpreter,'FontSize',FSize)

%title('Vegetation','interpreter','latex','FontSize',FSize)
%axes(ha(2));
%title('Soil','interpreter','latex','FontSize',FSize)
%axes(ha(3));
%title('Water','interpreter','latex','FontSize',FSize)

for i=1:P
  axes(ha(i));
  title(cood{i}(3:end), 'interpreter', interpreter, 'FontSize', FSize)
end

colormap(jet)

set(fh, 'Position', [0 0 500 800])

% uncomment to save the figure:
% savefig('RESULTS/abundTests.fig')
% print('RESULTS/abundTests.pdf','-dpdf')
% [~,~]=system(['pdfcrop RESULTS/abundTests.pdf RESULTS/abundTests.pdf']);
save_dir = 'results'
% octave fig
savefig([save_dir '/abundTests.fig'])
% in readable (pdf) format
print([save_dir '/abundTests.pdf'],'-dpdf','-painters')
% Crop using pdfcrop
[~,~]=system(['pdfcrop ' save_dir '/abundTests.pdf ' save_dir '/abundTests.pdf']);




% plot the estimated endmembers  -----------------------------
K_decim_spec = 10;
wavelenths = linspace(0.4,2.4,L);
Nplots = 1;
idxM   = [];
for i=1:length(algnames)
    if ~isnan(M_est{i}), Nplots = Nplots + 1; idxM = [idxM i]; end
end

ff = figure;
%[ha, pos] = tight_subplot(Nplots, 3, 0.01, 0.1, 0.1);
%[ha, pos] = tight_subplot(Nplots, 3, 0.05, 0.1, 0.1);

%[ha, pos] = tight_subplot(Nplots, 3, [0.01 0.06], 0.1, 0.1);
[ha, pos] = tight_subplot(Nplots, P, [0.01 0.06], 0.1, 0.1);

% When testing on real image(s), we don't have Mth, so just plot true (average) endmembers M_avg

for j=1:P
    axes(ha(j));
    %plot(wavelenths,squeeze(Mth(:,j,1:K_decim_spec:end)))
    plot(wavelenths,squeeze(M_avg(:,j)))
    xlim([min(wavelenths) max(wavelenths)])
    %ylim([0 1.15*max(max(squeeze(Mth(:,j,1:K_decim_spec:end))))])
    ylim([0 1.15*max(squeeze(M_avg(:,j)))])
    set(gca,'xtick',[])
end

for i=1:Nplots-1
    for j=1:P
        %axes(ha(j + (i-0)*3));
        axes(ha(j + (i-0)*P));
        plot(wavelenths,squeeze(M_est{idxM(i)}(:,j,1:K_decim_spec:end)))
        xlim([min(wavelenths) max(wavelenths)])
        ylim([0 1.15*max(max(squeeze(M_est{idxM(i)}(:,j,1:K_decim_spec:end))))])
        if i<Nplots-1, set(gca,'xtick',[]), end
    end 
    axes(ha(1 + (i-0)*P));
    ylabel(algnames{idxM(i)},'interpreter',interpreter,'FontSize',FSize)
end

axes(ha(1));
ylabel('Reference','interpreter',interpreter,'FontSize',FSize)

%title('Vegetation','interpreter','latex','FontSize',FSize)
%axes(ha(2));
%title('Soil','interpreter','latex','FontSize',FSize)
%axes(ha(3));
%title('Water','interpreter','latex','FontSize',FSize)

for i=1:P
  axes(ha(i));
  title(cood{i}(3:end), 'interpreter', interpreter, 'FontSize', FSize)
  %xlabel('Wavelength [$\mu$m]','interpreter',interpreter,'FontSize',FSize)
end

%axes(ha(end-2))
%xlabel('Wavelength [$\mu$m]','interpreter','latex','FontSize',FSize)
%axes(ha(end-1))
%xlabel('Wavelength [$\mu$m]','interpreter','latex','FontSize',FSize)
%axes(ha(end))
%xlabel('Wavelength [$\mu$m]','interpreter','latex','FontSize',FSize)

colormap(jet)

set(ff, 'Position', [0 0 500 600])

% uncomment to save the figure:
save_dir = 'results'
% octave fig
savefig([save_dir '/estimEMsTests.fig'])
% in readable (pdf) format
print([save_dir '/estimEMsTests.pdf'],'-dpdf','-painters')
% Crop using pdfcrop
[~,~]=system(['pdfcrop ' save_dir '/estimEMsTests.pdf ' save_dir '/estimEMsTests.pdf']);



