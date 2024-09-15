clc;clear;close all;
% Load Data
eq=68;
load(['./eq-' num2str(eq) '.mat'])
dn = d1;
% Band-Pass Filter
dt = 0.0005;
outF=das_bandpass(dn,dt,1e-3,250,6,6,0,0);
% Median Filter
d_bpmf=das_mf(outF,5,1,1);
% CWT Scale
wtmexh = cwtft2(d_bpmf,'wavelet','mexh','scales',1:0.5:10);
s=size(wtmexh.cfs);
out = wtmexh.cfs(:,:,1,s(4));
% Saving the CWT
save(['data-',num2str(eq)],'out','outF')