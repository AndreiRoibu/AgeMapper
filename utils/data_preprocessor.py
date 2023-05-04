import os
import h5py
import numpy as np
from fsl.data.image import Image
from fsl.utils.image.roi import roi
from data_utils import load_subjects_from_path, select_datasets_path, select_databases_path
from settings import Settings
from misc import create_folder
import timeit
from datetime import datetime

def construct_hdf5(data_parameters: dict, 
                   file_information: dict) -> None:
    
    """
    Constructs the HDF5 file
    
    Parameters:
    -----------
    data_parameters : dict
        Dictionary containing the data parameters
    file_information : dict
        Dictionary containing the file information

    Returns:
    --------
    None
    
    """

    # Read the subjects from the files!

    X_train_list_path, y_train_labels_path, y_train_ages_path, X_validation_list_path, y_validation_labels_path, y_validation_ages_path = select_datasets_path(data_parameters)

    train_subjects =  load_subjects_from_path(data_list=X_train_list_path)
    validation_subjects = load_subjects_from_path(data_list=X_validation_list_path)

    data_directory = data_parameters['data_directory']
    modality_flag = data_parameters['modality_flag']
    if modality_flag == 'T1_nonlinear':
        input_file = "T1/T1_brain_to_MNI.nii.gz"
    elif modality_flag == 'T1_linear':
        input_file = "T1/T1_brain_to_MNI_linear.nii.gz"
    else:
        print("ERROR - MODALITY NOT CURRENTLY SUPPORTED!")
    
    print('-> Processing training data:')

    train_input_volumes, train_target_labels, train_target_ages = load_datasets(subjects = train_subjects, 
                                                                                data_directory = data_directory, 
                                                                                input_file = input_file,
                                                                                modality_flag = modality_flag,
                                                                                test_age_labels = y_train_labels_path,
                                                                                test_ages = y_train_ages_path
                                                                                )

    print()
    print("The lenght of the training inputs is {} and the size of the elements is {}".format(len(train_input_volumes), train_input_volumes[len(train_input_volumes)-1].shape))
    print("The lenght of the training targets lables is {} and the size of the elements is {}".format(len(train_target_labels), train_target_labels[len(train_target_labels)-1].shape))
    print("The lenght of the training targets ages is {} and the size of the elements is {}".format(len(train_target_ages), train_target_ages[len(train_target_ages)-1].shape))

    write_hdf5(train_input_volumes, train_target_labels, train_target_ages , file_information, mode='train')

    del train_input_volumes, train_target_labels, train_target_ages                                                                                                 

    # Then, we'll do it for the validation data

    print('-> Processing validation data:')

    validation_input_volumes, validation_target_labels, validation_target_ages = load_datasets(subjects = validation_subjects, 
                                                                                                data_directory = data_directory, 
                                                                                                input_file = input_file,
                                                                                                modality_flag = modality_flag,
                                                                                                test_age_labels = y_validation_labels_path,
                                                                                                test_ages = y_validation_ages_path
                                                                                                )

    print()
    print("The lenght of the validation inputs is {} and the size of the elements is {}".format(len(validation_input_volumes), validation_input_volumes[len(validation_input_volumes)-1].shape))
    print("The lenght of the validation targets lables is {} and the size of the elements is {}".format(len(validation_target_labels), validation_target_labels[len(validation_target_labels)-1].shape))
    print("The lenght of the validation targets ages is {} and the size of the elements is {}".format(len(validation_target_ages), validation_target_ages[len(validation_target_ages)-1].shape))

    write_hdf5(validation_input_volumes, validation_target_labels, validation_target_ages , file_information, mode='validation')

    del validation_input_volumes, validation_target_labels, validation_target_ages


def write_hdf5(input_volumes: np.ndarray, 
               target_labels: np.ndarray, 
               target_ages: np.ndarray, 
               file_information: dict, 
               mode: str) -> None:
    
    """
    Writes the HDF5 file

    Parameters:
    -----------
    input_volumes : np.ndarray
        Input volumes
    target_labels : np.ndarray
        Target labels
    target_ages : np.ndarray
        Target ages
    file_information : dict
        Dictionary containing the file information
    mode : str
        Mode (train or validation)

    Returns:
    --------
    None

    """

    if os.path.exists(file_information[mode]['input_volumes']):
        os.remove(file_information[mode]['input_volumes'])

    if os.path.exists(file_information[mode]['target_labels']):
        os.remove(file_information[mode]['target_labels'])

    if os.path.exists(file_information[mode]['target_ages']):
        os.remove(file_information[mode]['target_ages'])

    if os.path.exists(file_information[mode]['target_labels']):
        os.remove(file_information[mode]['target_labels'])

    with h5py.File(file_information[mode]['input_volumes'], 'w') as data_handle:
        data_handle.create_dataset('input_volumes', data=input_volumes)
    
    with h5py.File(file_information[mode]['target_labels'], 'w') as data_handle:
        data_handle.create_dataset('target_labels', data=target_labels)

    with h5py.File(file_information[mode]['target_ages'], 'w') as data_handle:
        data_handle.create_dataset('target_ages', data=target_ages)


def load_datasets(subjects: list, 
                  data_directory: str, 
                  input_file: str, 
                  modality_flag: str, 
                  test_age_labels: str, 
                  test_ages: str) -> tuple:
    
    """
    Loads the datasets

    Parameters:
    -----------
    subjects : list
        List of subjects
    data_directory : str
        Data directory
    input_file : str
        Input file
    modality_flag : str
        Modality flag
    test_age_labels : str
        Test age labels
    test_ages : str
        Test ages

    Returns:
    --------
    input_volumes : np.ndarray
        Input volumes
    target_labels : np.ndarray
        Target labels
    target_ages : np.ndarray
        Target ages

    """
    
    print("Loading and pre-processing data")

    input_volumes, target_labels, target_ages = [], [], []

    len_subjects = len(subjects)

    for index, subject in enumerate(subjects):

        input_volume, target_label, target_age = load_and_preprocess(subject, index, data_directory, input_file, modality_flag, test_age_labels, test_ages)

        input_volumes.append(input_volume)
        target_labels.append(target_label)
        target_ages.append(target_age)

        print("\r Processed {:.3f}%: {}/{} inputs".format((index+1)/len_subjects * 100.0, len(input_volumes), len_subjects), end='')

    return input_volumes, target_labels, target_ages


def load_and_preprocess(subject: str, 
                        index: int, 
                        data_directory: str, 
                        input_file: str, 
                        modality_flag: str, 
                        test_age_labels: str, 
                        test_ages: str) -> tuple:
    
    """
    Loads and preprocesses the data

    Parameters:
    -----------
    subject : str
        Subject
    index : int
        Index
    data_directory : str
        Data directory
    input_file : str
        Input file
    modality_flag : str
        Modality flag
    test_age_labels : str
        Test age labels
    test_ages : str
        Test ages

    Returns:
    --------
    input_volume : np.ndarray
        Input volume
    target_label : np.ndarray
        Target label
    target_age : np.ndarray
        Target age

    """


    input_volume, target_label, target_age = load_data(subject, index, data_directory, input_file, test_age_labels, test_ages)
    input_volume  = preprocess(input_volume, modality_flag)

    return np.float32(input_volume), np.float32(target_label), np.float32(target_age)


def load_data(subject: str, 
              index: int, 
              data_directory: str, 
              input_file: str, 
              test_age_labels: str, 
              test_ages: str) -> tuple:
    
    """
    Loads the data

    Parameters:
    -----------
    subject : str
        Subject
    index : int
        Index
    data_directory : str
        Data directory
    input_file : str
        Input file
    test_age_labels : str
        Test age labels
    test_ages : str
        Test ages

    Returns:
    --------
    input_volume : np.ndarray
        Input volume
    target_label : np.ndarray
        Target label
    target_age : np.ndarray
        Target age

    """


    input_path = os.path.join(os.path.expanduser("~"), data_directory, subject, input_file)
    input_volume = roi(Image(input_path),((10,170),(12,204),(0,160))).data
    input_volume = np.float32(input_volume)

    target_label = np.load(test_age_labels)[index]
    target_age = np.load(test_ages)[index]

    return input_volume, target_label, target_age


def preprocess(input_volume: np.ndarray, 
               modality_flag: str) -> np.ndarray:
    
    """
    Preprocesses the data

    Parameters:
    -----------
    input_volume : np.ndarray
        Input volume
    modality_flag : str
        Modality flag
        
    Returns:
    --------
    input_volume : np.ndarray
        Input volume

    """

    if modality_flag == 'T1_nonlinear' or modality_flag == 'T1_linear':
        input_volume = input_volume / input_volume.mean()
        if modality_flag == 'T1_nonlinear':
            input_volume = input_volume / 5.550105992097339
        else:
            input_volume = input_volume / 5.678573626611138
        input_volume = np.float32(input_volume)
    else:
        print("ERROR - MODALITY NOT YET SUPPORTED!")
    
    return input_volume


if __name__ == "__main__":

    print('Started Data Generation!')

    start_time = datetime.now()
    start_time2 = timeit.default_timer()
    print('Started At: {}'.format(start_time))

    settings = Settings('utils/data_preprocessor.ini')
    data_parameters = settings['DATA']
    data_folder_name = data_parameters['data_folder_name']

    create_folder(data_folder_name)

    X_train, y_train_labels, y_train_ages, X_validation, y_validation_labels, y_validation_ages = select_databases_path(data_parameters)

    file_information = {
                        'train': {"input_volumes" : os.path.join(data_folder_name, X_train),
                                  "target_labels" : os.path.join(data_folder_name, y_train_labels),
                                  "target_ages" : os.path.join(data_folder_name, y_train_ages),
                                  },
                        'validation': {"input_volumes" : os.path.join(data_folder_name, X_validation),
                                       "target_labels" : os.path.join(data_folder_name, y_validation_labels),
                                       "target_ages" : os.path.join(data_folder_name, y_validation_ages),
                                      }
                        }

    construct_hdf5(data_parameters, file_information)

    print('Completed Data Generation!')

    end_time = datetime.now()
    end_time2 = timeit.default_timer()
    print('Completed At: {}'.format(end_time))
    print('Training Duration: {}'.format(end_time - start_time))
    print('Training Duration 2: {}'.format(end_time2 - start_time2))

