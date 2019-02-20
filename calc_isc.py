import numpy as np
import matplotlib.pyplot as plt
import configparser, os, re
from scipy.stats import zscore,pearsonr
import nibabel as nib
import time
#%%
# Create simple simulated data with high intersubject correlation
def simulated_timeseries(n_subjects, n_TRs, n_voxels=1, noise=1):
    signal = np.random.randn(n_TRs, n_voxels // 100)
    data = [zscore(np.repeat(signal, 100, axis=1) +
                 np.random.randn(n_TRs, n_voxels) * noise,
                 axis=0)
          for subject in np.arange(n_subjects)]
    
    return data

def plot_isc(c,participant,foutroot):
    """Given a 2D matrix c creates a 2D colormap from -1 to 1"""

    plt.figure(figsize = (5,4))
    plt.imshow(c, vmin = -1, vmax = 1, cmap = 'PRGn')
    plt.xlabel('# Voxels')
    plt.ylabel('# Voxels')
    plt.colorbar()
    plt.title('Leave-one-out ISC // Participant {}'.format(participant))
    fout = foutroot + 'c_{}.{}'.format(participant,fig_fmt)        
    plt.savefig(fout,dpi = 300,fmt=fig_fmt)
    plt.close('all')
    
    return

def fill_n(string,n):
    
    try:
        beg, temp = string.split('{')
        to_rep, final = temp.split('}')
        num_digits = len(to_rep)

        result = beg + '{num:0{ndigits}}'.format(num=n,ndigits=num_digits) + final
        
    except:
        result = string
        
    return result

def make_file_list(subjects):
    
    file_list = []
    n_true = 0
    
    for n in subjects:
    
        fin = froot + fill_n(fld_example,n) + fill_n(input_fname,n)
        
        if os.path.isfile(fin):
            file_list.append(fin)
            n_true += 1
        else:
            print('Warning!!! No file found in %s' % fin)
            
    print('Ready to load data for {} participants'.format(n_true))
    
    return file_list


def import_input_file(fin,sub_sample = False):
    
    input_ext = fin.split('.')[-1]
    
    if input_ext == 'mat':
        
        print('Please input ,nii files.')
        
    elif input_ext == 'nii':
        
        img = nib.load(fin)            
        data = img.get_fdata()
        hdr = img.header     
        aff = img.affine
        
        nx,ny,nz, n_TRs = data.shape
        n_voxels = np.prod([nx,ny,nz])
        
        # data = data.reshape((n_voxels,n_TRs)).T # TODO double check that this transpose doesn't hurt
        
        # If you want to play with less data
        if sub_sample == True:
            
            n_sampled_voxels = 100
            r_indices = np.arange(n_sampled_voxels)
#                r_indices = np.random.permutation(n_voxels)[:n_sampled_voxels]
            v_indices = np.sort(r_indices)
        
            data = data[:,v_indices]
    
    return hdr,aff,data

def compute_participant_average(subjects):

    file_list = make_file_list(subjects)
    n_files = len(file_list)

    for k,fin in enumerate(file_list):

        # If it's the first I need to initialize the average
        if k == 0:
            hdr,aff,average = import_input_file(fin)
        else:
            average += import_input_file(fin)[-1]
            
        print('Averaging... %d out of %d' % (k +1, n_files))

    average = average / float(n_files)

    fout = foutroot + 'subjects_average.nii'
    img = nib.Nifti1Image(average, aff, header = hdr)
    img.to_filename(fout)

    
    return average


def leave_one_out_isc(subjects):
    

    n_subjects = len(subjects)
    file_list = make_file_list(subjects)

    # loop trough every subject
    for s,fin in zip(subjects,file_list): 

        print('Computing ISC for participant %d / %d' % (s, n_subjects))

        hdr,aff,data = import_input_file(fin)
       
        # Prepare one array for output
        corr_data = np.zeros((nx,ny,nz),dtype = float)
                
        # Calculate the  correlation for every voxel
        for vx in range(nx):
            for vy in range(ny):
                for vz in range(nz):
                    # if there is some variation in the data
                    if np.diff(data[vx,vy,vz,:]).any() != 0:
                        # Compute the correlation coefficient
                        corr_data[vx,vy,vz] = pearsonr(data_average[vx,vy,vz,:] - ( data[vx,vy,vz,:] / (1.0 * n_subjects) ),
                                                         data[vx,vy,vz,:])[0]
                    else: # Otherwise set it to zero
                        corr_data[vx,vy,vz] = 0
                    
    
        # after correlating we can free the memory from the data
        del data
        
        # And save it as a nifti file
        fout = foutroot + 'leaveoneout_c_S{:02}.nii'.format(s)
        img = nib.Nifti1Image(corr_data, aff, header = hdr)
        nib.save(img,fout)
        
        # Save also the Fisher z-transformation
        fout = foutroot + 'leaveoneout_z_S{:02}.nii'.format(s)
        img = nib.Nifti1Image(np.arctanh(corr_data), aff, header = hdr)
        img.to_filename(fout)



    return

    
#%%
########## MAIN CODE ##########
if __name__ == '__main__':
    


    ### IMPORT INPUT DATA ###
    config = configparser.ConfigParser()
    config.read('settings.ini')
    froot = config['INPUT']['Folder with input data']
    N_subjects = int(config['INPUT']['N participants'])
    fld_example = config['INPUT']['Folder hierarchy']
    input_fname = config['INPUT']['File name']
    input_ext = input_fname.split('.')[-1]
 
    # Prepare parameters for output
    subjects = np.arange(N_subjects) + 1

    foutroot = froot + 'ISC_%02d-%02d/' % (subjects[0],subjects[-1])
    if not os.path.exists(foutroot):
        os.makedirs(foutroot)
        
        
    # Load the average among participants or compute it
    start = time.time()
    
    try:
        img = nib.load(foutroot + 'subjects_average.nii')
        print('Loading data average from %ssubjects_average.nii' % foutroot)
        data_average = img.get_fdata()
    except:
        print('No file with average data found. Computing the average now.')
        data_average = compute_participant_average(subjects)
        
    nx,ny,nz, n_TRs = data_average.shape
    
    avg_done = time.time()
    T_avg = avg_done - start
    print('Average computed in %d min and %d s' % (int(T_avg/60),int(np.mod(T_avg,60))))
    
    # Compute the leave one out isc
    leave_one_out_isc(subjects)
    
    isc_done = time.time()
    T_isc = isc_done - avg_done
    print('ISC computed in %d min and %d s' % (int(T_isc/60),int(np.mod(T_isc,60))))
    
    # byeee