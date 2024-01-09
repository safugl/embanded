rootdir = fileparts(mfilename('fullfile'));
addpath(fullfile(rootdir,'matlab'))
addpath(fullfile(rootdir,'matlab','functions'))
addpath(fullfile(rootdir,'examples','matlab'))
addpath(fullfile(rootdir,'examples','matlab','func'))

fprintf('\n ======================================================================')
fprintf('\n Setting up embanded on MATLAB version: %s \n' ,version)