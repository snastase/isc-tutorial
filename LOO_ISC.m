% This script takes all the files in inDir that satisfy a certain searchstring 
% then it calculates the average time course and all the leave-one-out correlations and saves them in outDir.
% the script uses the nifti toolbox, so make sure to install it in your path see https://nl.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image

% make sure all the 4D files are in the same image space and have the same
% number of time-points or you'll get an error

% script written by Christian Keysers

inDir='/Users/christiankeysers/Dropbox (Social Brain Lab)/idisk/CurrentManuscripts/ISCtoolsoftrade/LorenzoPy/ExampleData/InputData'; %set this to where all your 4D files are
outDir='/Users/christiankeysers/Dropbox (Social Brain Lab)/idisk/CurrentManuscripts/ISCtoolsoftrade/LorenzoPy/ExampleData/OutputData'; %create that directory before running the script
conditions=dir(strcat([inDir,'/','swa*.nii'])); %swa*.nii is the searchstring for your 4D files.

n=length(conditions)

%% calculates the overall sum. 
% because most PCs have limited memory, we do not load all the 4D files, but build up the mean by simply opening a file
% adding it to a running sum, and then opening the next file into the same
% variable. Because we later use correlations, where the mean or the sum
% has the same effect, we use the sum.

sum=load_nii(strcat(conditions(1).folder,'/',conditions(1).name)); %this will setup sum to contain the first image only

for i=2:n
    tmp=load_nii(strcat(conditions(i).folder,'/',conditions(i).name));
    sum.img=sum.img+tmp.img;
end

%% saves the sum for future use
save_nii(sum,strcat([outDir '/' 'sum.nii']));
sd=std(sum.img,[],4);

%% gets the sizes of the 4D files to step through all the voxels
[x,y,z,t]=size(sum.img);
%% calculate for each image, the correlation with others and the fisher-z-transformed maps and saves into outDir
% this script uses a simple trick: the average of other participants is not
% recalculated using all other participants, but simply as the sum - this
% particular participant. This allows the process to only need the sum and
% a particular participants 4D file to be open simultaneously
rho=load_nii(strcat(conditions(1).folder,'/',conditions(1).name),1); % to make sure that the correlation images have the same header as the input images, I load the first volume of an image into rho, then zero the values in the .img
f = waitbar(0,'Please wait...');
for subj=1:n
    waitbar(subj/n,f,strcat(['starting to process subject ',num2str(subj,'%02.f')]));
    tmp=load_nii(strcat(conditions(subj).folder,'/',conditions(subj).name));
    rho.img=NaN(x,y,z);

    for xi=1:x
        for yi=1:y
            for zi=1:z
                if not(sd(xi,yi,zi)==0)  % this excludes all voxels with zero variance to save time
                  rho.img(xi,yi,zi)=corr(squeeze(tmp.img(xi,yi,zi,:)),squeeze(sum.img(xi,yi,zi,:))-squeeze(tmp.img(xi,yi,zi,:)));
                end
            end
        end
    end

    save_nii(rho,strcat([outDir '/' 'LOO_R_' num2str(subj,'%02.f') '.nii']));
    fisherz=rho;
    fisherz.img=atanh(rho.img);
    save_nii(fisherz,strcat([outDir '/' 'LOO_Z_' num2str(subj,'%02.f') '.nii']));
end


