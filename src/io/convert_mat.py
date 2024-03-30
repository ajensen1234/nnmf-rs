import scipy.io 
import pandas as pd
import numpy as np
import os

class NMF_Data_Loader():
    def __init__(
            self,
            path_to_mat: str,
            local_data_dir: str
    ):
        self.path_to_mat = path_to_mat
        self.local_data_dir = local_data_dir
        self.patient_id = os.path.splitext(os.path.split(path_to_mat)[1])[0].split("_")[0]
        print(self.patient_id)

    def generate_all_data(self):
        num_synergy_index = lambda i: i-1
        data_mat = scipy.io.loadmat(self.path_to_mat)
        self.nmfL = data_mat["synergy_nmfL"]
        self.nmfR = data_mat["synergy_nmfR"]

        W_L = self.nmfL['W']
        W_R = self.nmfR['W']

        C_L = self.nmfL['C']
        C_R = self.nmfR['C']

        emgOrig_L = self.nmfL['emgOrig'][0,0]
        emgOrig_R = self.nmfR['emgOrig'][0,0]

        self.num_muscles_L = emgOrig_L.shape[0]
        self.num_muscles_R = emgOrig_R.shape[0]

        # now that we have all the data, we are going to create some folders based on this

        # we have our inputs and our verification
        # start by creating a CSV file that represents the desired input
        emg_fname = lambda pat_dir, side: pat_dir + "/" + self.patient_id + "_EMG_" + side + ".csv"
        # check if data exists
        pat_dir = self.local_data_dir + "/" + self.patient_id
        if not os.path.isdir(pat_dir):
            os.makedirs(pat_dir)
        # create validation directory
        if not os.path.isdir(pat_dir + "/val"):
            os.makedirs(pat_dir + "/val")
        
        # save CSV to numpy array
        
        np.savetxt(emg_fname(pat_dir, "R"), emgOrig_R, fmt = '%.4g', delimiter = ",")
        np.savetxt(emg_fname(pat_dir, "L"), emgOrig_L, fmt = '%.4g', delimiter = ",")

        # now we want to loop through the total number of muscles in order to generate validation sets
        w_fname = lambda pat_dir, side, num_syn: pat_dir + "/val/W_" + str(num_syn) + "_" + side +  ".csv"
        c_fname = lambda pat_dir, side, num_syn: pat_dir + "/val/C_" + str(num_syn) + "_" + side +  ".csv"
        for num_syn in range(1,self.num_muscles_R):
            # thresholding
            W = self.nmfR['W'][0,num_synergy_index(num_syn)]
            W[np.abs(W) < 1e-10] = 0
            np.savetxt(w_fname(pat_dir,"R", num_syn), W, fmt = '%.4g',delimiter = ",")

            C = self.nmfR['C'][0,num_synergy_index(num_syn)]
            np.savetxt(c_fname(pat_dir,"R", num_syn), C, fmt = '%.4g', delimiter=",")
        
        # now do the left side
        for num_syn in range(1,self.num_muscles_L):
            # thresholding
            W = self.nmfL['W'][0,num_synergy_index(num_syn)]
            W[np.abs(W) < 1e-10] = 0
            np.savetxt(w_fname(pat_dir,"L", num_syn), W, fmt = '%.4g',delimiter = ",")

            C = self.nmfL['C'][0,num_synergy_index(num_syn)]
            np.savetxt(c_fname(pat_dir,"L", num_syn), C, fmt = '%.4g', delimiter=",")