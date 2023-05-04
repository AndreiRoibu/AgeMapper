import os
import numpy as np
import torch
import torch.utils.data as data
import h5py
import pandas as pd
from fsl.data.image import Image
from fsl.utils.image.roi import roi
from scipy.stats import norm
import nibabel as nib

class DataMapper(data.Dataset):

    """
    Data Mapper Class
    """

    def __init__(self, 
                 X: np.ndarray, 
                 y_labels: np.ndarray, 
                 y_ages: np.ndarray) -> None:
        
        """
        Data Mapper Class Initialiser

        Parameters:
        -----------
        X : np.ndarray
            Input volumes
        y_labels : np.ndarray
            Target labels
        y_ages : np.ndarray
            Target ages

        Returns:
        --------
        None

        """
        
        self.X = X
        self.y_labels = y_labels
        self.y_ages = y_ages
        
    def __getitem__(self, index):
        X_volume = torch.from_numpy(self.X[index])
        y_age = np.array(self.y_ages[index])

        return X_volume, y_age

    def __len__(self):
        return len(self.y_ages)


def select_databases_path(data_parameters: dict) -> tuple:
    """
    Function to select the database paths based on the data parameters

    Parameters:
    -----------
    data_parameters : dict
        Dictionary containing the data parameters

    Returns:
    --------
    X_train : str
        Path to the training input volumes
    y_train_ages : str
        Path to the training target ages
    X_validation : str
        Path to the validation input volumes
    y_validation_ages : str
        Path to the validation target ages

    """
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']

    X_train = data_parameters['male_train_database']
    y_train_ages = data_parameters['male_train_ages_database']
    X_validation = data_parameters['male_validation_database']
    y_validation_ages = data_parameters['male_validation_ages_database']

    if dataset_sex == 'female':
        X_train = "fe" + X_train
        y_train_ages = "fe" + y_train_ages
        X_validation = "fe" + X_validation
        y_validation_ages = "fe" + y_validation_ages

    if data_parameters['modality_flag'] == 'T1_nonlinear':
        X_train += "_T1_nonlinear"
        y_train_ages += "_T1_nonlinear"
        X_validation += "_T1_nonlinear"
        y_validation_ages += "_T1_nonlinear"
    elif data_parameters['modality_flag'] == 'T1_linear':
        X_train += "_T1_linear"
        y_train_ages += "_T1_linear"
        X_validation += "_T1_linear"
        y_validation_ages += "_T1_linear"  

    if dataset_size == 'small':
        X_train += "_small.h5"
        y_train_ages += "_small.h5"
        X_validation += "_small.h5"
        y_validation_ages += "_small.h5"
    else:
        X_train += ".h5"
        y_train_ages += ".h5"
        X_validation += ".h5"
        y_validation_ages += ".h5"

    return X_train, y_train_ages, X_validation, y_validation_ages


def get_datasets(data_parameters: dict) -> tuple:
    """
    Function to get the datasets based on the data parameters

    Parameters:
    -----------
    data_parameters : dict
        Dictionary containing the data parameters

    Returns:
    --------
    DataMapper
        Training dataset
    DataMapper
        Validation dataset

    """


    key_X = 'input_volumes'
    key_y_ages = 'target_ages'

    X_train, y_train_ages, X_validation, y_validation_ages = select_databases_path(data_parameters)

    X_train_data = h5py.File(os.path.join(data_parameters["data_folder_name"], X_train), 'r')
    y_train_ages_data = h5py.File(os.path.join(data_parameters["data_folder_name"], y_train_ages), 'r')
    
    X_validation_data = h5py.File(os.path.join(data_parameters["data_folder_name"], X_validation), 'r')
    y_validation_ages_data = h5py.File(os.path.join(data_parameters["data_folder_name"], y_validation_ages), 'r')

    if data_parameters['load_full_data_into_memory'] == True:
        return (
            DataMapper( X_train_data[key_X][()], y_train_ages_data[key_y_ages][()] ),
            DataMapper( X_validation_data[key_X][()], y_validation_ages_data[key_y_ages][()] ),
        )
    else:
        return (
                DataMapper( X_train_data[key_X], y_train_ages_data[key_y_ages] ),
                DataMapper( X_validation_data[key_X], y_validation_ages_data[key_y_ages] )
        )


class DynamicDataMapper(data.Dataset):

    """
    Data Mapper Class
    
    This class is used to load the data dynamically. This is useful when the data is too large to be loaded into memory.
    This is the default DataMapper class.
    """

    def __init__(self, 
                 X_paths: list,  
                 y_ages: np.ndarray, 
                 modality_flag: str, 
                 scale_factor: float, 
                 resolution: str, 
                 rsfmri_volume: int = None, 
                 shift_transformation: bool = False,
                 mirror_transformation: bool = False) -> None:
        
        """
        Data Mapper Class Initialiser

        Parameters:
        -----------
        X_paths : list
            List of paths to the input volumes
        y_ages : np.ndarray
            Target ages
        modality_flag : str
            Modality flag
        scale_factor : float
            Scale factor to be applied to the input volumes
        resolution : str
            Resolution of the input volumes
        rsfmri_volume : int
            rsfMRI volume number
        shift_transformation : bool
            Whether to apply shift transformation or not
        mirror_transformation : bool
            Whether to apply mirror transformation or not

        Returns:
        --------
        None
            
        """

        self.X_paths = X_paths # List where all the file paths are already in the deisred order
        self.y_ages = y_ages
        self.scale_factor = scale_factor
        self.shift_transformation = shift_transformation
        self.mirror_transformation = mirror_transformation

        self.rsfmri_volume = rsfmri_volume

        non_deterministic_modalities = ['T1_nonlinear', 'T1_linear', 'T2_nonlinear']
        if modality_flag in non_deterministic_modalities:
            self.non_deterministic_modality = True
        else:
            self.non_deterministic_modality = False

        if resolution == '2mm':
            self.crop_values = [5, 85, 6, 102, 0, 80]
        else:
            self.crop_values = [10, 170, 12, 204, 0, 160]
        
    def __getitem__(self, index):

        if self.rsfmri_volume != None:
            X_volume = np.array(nib.load(self.X_paths[index]).dataobj)[:,:,:,self.rsfmri_volume]
        else:
            X_volume = np.array(nib.load(self.X_paths[index]).dataobj)

        X_volume = X_volume[self.crop_values[0]:self.crop_values[1],
                            self.crop_values[2]:self.crop_values[3], 
                            self.crop_values[4]:self.crop_values[5]]

        if self.non_deterministic_modality == True:
            X_volume = X_volume / X_volume.mean()

        X_volume = X_volume / self.scale_factor

        if self.mirror_transformation==True:
            prob = np.random.rand(1)
            if prob < 0.5:
                X_volume = np.flip(X_volume,0)

        if self.shift_transformation==True:
            x_shift, y_shift, z_shift = np.random.randint(-2,3,3)
            X_volume = np.roll(X_volume,x_shift,axis=0)
            X_volume = np.roll(X_volume,y_shift,axis=1)
            X_volume = np.roll(X_volume,z_shift,axis=2)
            if z_shift < 0:
                X_volume[:,:,z_shift:] = 0

        X_volume = torch.from_numpy(X_volume)
        y_age = np.array(self.y_ages[index])

        return X_volume, y_age

    def __len__(self):
        return len(self.X_paths)


def select_datasets_path(data_parameters: dict) -> tuple:
    """
    Function to select the datasets paths based on the data parameters

    Parameters:
    -----------
    data_parameters : dict
        Dictionary containing the data parameters

    Returns:
    --------    
    X_train_list_path : str
        Path to the training input volumes
    y_train_ages_path : str
        Path to the training target ages
    X_validation_list_path : str
        Path to the validation input volumes
    y_validation_ages_path : str
        Path to the validation target ages

    """
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']
    data_folder_name = data_parameters['data_folder_name']

    X_train_list_path = data_parameters['male_train']
    y_train_ages_path = data_parameters['male_train_age']
    X_validation_list_path = data_parameters['male_validation']
    y_validation_ages_path = data_parameters['male_validation_age']

    if dataset_sex == 'female':
        X_train_list_path = "fe" + X_train_list_path
        y_train_ages_path = "fe" + y_train_ages_path
        X_validation_list_path = "fe" + X_validation_list_path
        y_validation_ages_path = "fe" + y_validation_ages_path

    if dataset_size == 'small':
        X_train_list_path += "_small.txt"
        y_train_ages_path += "_small.npy"
        X_validation_list_path += "_small.txt"
        y_validation_ages_path += "_small.npy"
    elif dataset_size == 'tiny':
        X_train_list_path += "_tiny.txt"
        y_train_ages_path += "_tiny.npy"
        X_validation_list_path += "_tiny.txt"
        y_validation_ages_path += "_tiny.npy"
    else:
        X_train_list_path += ".txt"
        y_train_ages_path += ".npy"
        X_validation_list_path += ".txt"
        y_validation_ages_path += ".npy"

    if dataset_size == 'han':
        X_train_list_path = "train_han.txt"
        y_train_ages_path = "train_age_han.npy"
        X_validation_list_path = "validation_han.txt"
        y_validation_ages_path = "validation_age_han.npy"

    if dataset_size == 'everything':
        X_train_list_path = "train_everything.txt"
        y_train_ages_path = "train_age_everything.npy"
        X_validation_list_path = "validation_everything.txt"
        y_validation_ages_path = "validation_age_everything.npy"

    if 'small' in dataset_size and dataset_size!='small':
        # ATTENTION! Cross Validation only enabled for male subjects at the moment!
        print('ATTENTION! CROSS VALIDATION DETECTED. This will only work for small male subject datasets ATM!')
        X_train_list_path = data_parameters['male_train'] + '_' + dataset_size + '.txt'
        y_train_ages_path = data_parameters['male_train_age'] + '_' + dataset_size + '.npy'
        X_validation_list_path = data_parameters['male_validation'] + '_' + dataset_size + '.txt'
        y_validation_ages_path = data_parameters['male_validation_age'] + '_' + dataset_size + '.npy'

    X_train_list_path = os.path.join(data_folder_name, X_train_list_path)
    y_train_ages_path = os.path.join(data_folder_name, y_train_ages_path)
    X_validation_list_path = os.path.join(data_folder_name, X_validation_list_path)
    y_validation_ages_path = os.path.join(data_folder_name, y_validation_ages_path)

    return X_train_list_path, y_train_ages_path, X_validation_list_path, y_validation_ages_path


def get_datasets_dynamically(data_parameters: dict) -> tuple:
    """
    Function to get the datasets based on the data parameters

    Parameters:
    -----------
    data_parameters : dict
        Dictionary containing the data parameters

    Returns:
    --------
    DynamicDataMapper
        Training dataset
    DynamicDataMapper
        Validation dataset
    str
        Resolution of the input volumes

    """

    X_train_list_path, y_train_ages_path, X_validation_list_path, y_validation_ages_path = select_datasets_path(data_parameters)

    data_directory = data_parameters['data_directory']
    modality_flag = data_parameters['modality_flag']
    scaling_values_simple = pd.read_csv(data_parameters['scaling_values'], index_col=0)

    scale_factor = scaling_values_simple.loc[modality_flag].scale_factor
    resolution = scaling_values_simple.loc[modality_flag].resolution
    data_file = scaling_values_simple.loc[modality_flag].data_file

    modality_flag_split = modality_flag.rsplit('_', 1)
    if modality_flag_split[0] == 'rsfmri':
        rsfmri_volume = int(modality_flag_split[1])
    else:
        rsfmri_volume = None

    shift_transformation = data_parameters['shift_transformation']
    mirror_transformation = data_parameters['mirror_transformation']

    X_train_paths, _ = load_file_paths(X_train_list_path, data_directory, data_file)
    X_validation_paths, _ = load_file_paths(X_validation_list_path, data_directory, data_file)
    y_train_ages = np.load(y_train_ages_path)
    y_validation_ages = np.load(y_validation_ages_path)

    print('****************************************************************')
    print("DATASET INFORMATION")
    print('====================')
    print("Modality Name: ", modality_flag)
    if rsfmri_volume != None:
        print("rsfMRI Volume: ", rsfmri_volume)
    print("Resolution: ", resolution)
    print("Scale Factor: ", scale_factor)
    print("Data File Path: ", data_file)
    print('****************************************************************')

    return (
        DynamicDataMapper( X_paths=X_train_paths, y_ages=y_train_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, rsfmri_volume=rsfmri_volume,
                            shift_transformation=shift_transformation, mirror_transformation=mirror_transformation),
        DynamicDataMapper( X_paths=X_validation_paths, y_ages=y_validation_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, rsfmri_volume=rsfmri_volume,
                            shift_transformation=False, mirror_transformation=False),
        resolution
    )


def load_file_paths(data_list: str, 
                    data_directory: str, 
                    mapping_data_file: str) -> tuple:
    """
    Function to load the file paths from a data list

    Parameters:
    -----------
    data_list : str
        Path to the data list
    data_directory : str
        Path to the data directory
    mapping_data_file : str
        Name of the mapping data file

    Returns:
    --------
    file_paths : list
        List of file paths
    volumes_to_be_used : list
        List of volumes to be used

    """


    volumes_to_be_used = load_subjects_from_path(data_list)

    file_paths = [os.path.join(data_directory, volume, mapping_data_file) for volume in volumes_to_be_used]

    return file_paths, volumes_to_be_used 


def load_subjects_from_path(data_list: str) -> list:
    """
    Function to load the subjects from a data list

    Parameters:
    -----------
    data_list : str
        Path to the data list

    Returns:
    --------
    volumes_to_be_used : list
        List of volumes to be used

    """

    with open(data_list) as data_list_file:
        volumes_to_be_used = data_list_file.read().splitlines()

    return volumes_to_be_used


def load_and_preprocess_evaluation(file_path: str, 
                                   modality_flag: str, 
                                   resolution: str, 
                                   scale_factor: float, 
                                   rsfmri_volume: int = None) -> np.ndarray:
    """
    Function to load and preprocess the evaluation data

    Parameters:
    -----------
    file_path : str
        Path to the file
    modality_flag : str
        Modality flag
    resolution : str
        Resolution of the input volumes
    scale_factor : float
        Scale factor to be applied to the input volumes
    rsfmri_volume : int
        rsfMRI volume number

    Returns:
    --------
    volume : np.ndarray
        Preprocessed volume

    """
    

    non_deterministic_modalities = ['T1_nonlinear', 'T1_linear', 'T2_nonlinear']

    if rsfmri_volume != None:
        volume = np.array(nib.load(file_path).dataobj)[:,:,:,rsfmri_volume]
    else:
        volume = np.array(nib.load(file_path).dataobj)

    if resolution == '2mm':
        crop_values = [5, 85, 6, 102, 0, 80]
    else:
        crop_values = [10, 170, 12, 204, 0, 160]

    volume = volume[crop_values[0]:crop_values[1],
                    crop_values[2]:crop_values[3], 
                    crop_values[4]:crop_values[5]]

    if modality_flag in non_deterministic_modalities:
        volume = volume / volume.mean()

    volume = volume / scale_factor

    volume = np.float32(volume)

    return volume


def num2vec(label: np.ndarray, 
            bin_range: list = [44,84], 
            bin_step: int = 1, 
            std: int = 1):
    
    """
    Function to convert a label to a vector. This function was originally developed by Peng et. al. (2021) and was adapted for this project.

    Parameters:
    -----------
    label : np.ndarray
        Target label
    bin_range : list
        Bin range
    bin_step : int
        Bin step
    std : int
        Standard deviation

    Returns:
    --------
    bin_values : np.ndarray
        Bin values
    bin_centers : np.ndarray
        Bin centers

    """
    

    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if bin_length % bin_step != 0:
        print("Error: Bin range should be divisible by the bin step!")
        return None
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + bin_step/2.0 + bin_step * np.arange(bin_number)
    
    if std == 0:
        # Uniform Distribution Case
        label = np.array(label)
        bin_values = np.floor((label - bin_start)/bin_step).astype(int)
    elif std < 0:
        print("Error! The standard deviation (& variance) must be positive")
        return None
    else:
        bin_values = np.zeros((bin_number))
        for i in range(bin_number):
            x1 = bin_centers[i] - bin_step/2.0
            x2 = bin_centers[i] + bin_step/2.0
            cdfs = norm.cdf([x1, x2], loc=label, scale=std)
            bin_values[i] = cdfs[1] - cdfs[0]       

    return bin_values, bin_centers