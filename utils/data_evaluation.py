import os
import numpy as np
import torch
import torch.nn as nn
import logging
import utils.data_utils as data_utils
from utils.misc import create_folder
import pandas as pd
from AgeMapper import AgeMapper
from collections import OrderedDict

log = logging.getLogger(__name__)
MSELoss = nn.MSELoss()

def evaluate_data(trained_model_path: str,
                     data_directory: str,
                     data_list: str,
                     prediction_output_path: str,
                     prediction_output_statistics_name: str,
                     modality_flag: str,
                     control: str,
                     device: str,
                     test_ages: str,
                     dataset_sex: str,
                     scaling_values: str,
                     exit_on_error: bool = False,) -> None:
    
    """ Evaluate Data Function
    This function evaluates the data and generates the output statistics
    
    Parameters:
    -----------
    trained_model_path : str
        Path to the trained model
    data_directory : str
        Path to the data directory
    data_list : str
        Path to the data list
    prediction_output_path : str
        Path to the prediction output
    prediction_output_statistics_name : str
        Name of the prediction output statistics
    modality_flag : str
        Modality flag
    control : str
        Control
    device : str
        Device
    test_ages : str 
        Path to the test ages

    Returns:
    --------
    None

    """

                     
    log.info("Started Evaluation. Check tensorboard for plots (if a LogWriter is provided)")

    
    with open(data_list) as data_list_file:
        volumes_to_be_used = data_list_file.read().splitlines()


    scaling_values_simple = pd.read_csv(scaling_values, index_col=0)

    scale_factor = scaling_values_simple.loc[modality_flag].scale_factor
    resolution = scaling_values_simple.loc[modality_flag].resolution
    mapping_data_file = scaling_values_simple.loc[modality_flag].data_file

    modality_flag_split = modality_flag.rsplit('_', 1)
    if modality_flag_split[0] == 'rsfmri':
        rsfmri_volume = int(modality_flag_split[1])
    else:
        rsfmri_volume = None

    model = AgeMapper(resolution=resolution,
                      )

    trained_model = torch.load(trained_model_path,map_location=torch.device(device))

    if hasattr(trained_model, 'state_dict'):
        model_sd = trained_model.state_dict()
        if torch.cuda.device_count()>1 or device=='cpu':
            correct_state_dict = {}
            for key in model_sd.keys():
                if key.startswith('module.'):
                    new_key = key.replace('module.', "")
                    correct_state_dict[new_key] = model_sd[key]
                else:
                    correct_state_dict[key] = model_sd[key]
            correct_state_dict = OrderedDict(correct_state_dict)
            del model_sd
        model.load_state_dict(correct_state_dict)
    else:
        model.load_state_dict(trained_model)

    del trained_model

    if torch.cuda.is_available() == True and device!='cpu':
        model.cuda(device)
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = 'False'

    model.eval()

    # Create the prediction path folder if this is not available

    create_folder(prediction_output_path)

    # Initiate the evaluation

    log.info("Evaluation Started")

    file_paths, volumes_to_be_used = data_utils.load_file_paths(data_list, data_directory, mapping_data_file)

    if control == 'mean':
        prediction_output_statistics_name = "output_statistics_mean_target.csv"
    elif control == 'null':
        prediction_output_statistics_name = "output_statistics_null_target.csv"

    output_statistics = {}
    output_statistics_path = os.path.join(prediction_output_path, prediction_output_statistics_name)

    with torch.no_grad():

        for volume_index, file_path in enumerate(file_paths):
            try:
                print("\r Mapping Volume {:.3f}%: {}/{} test subjects".format((volume_index+1)/len(file_paths) * 100.0, volume_index+1, len(file_paths)), end='')

                # Generate volume & header

                subject = volumes_to_be_used[volume_index]

                output_age = _generate_age(file_path,
                                        model,
                                        device,
                                        cuda_available,
                                        modality_flag,
                                        resolution,
                                        scale_factor,
                                        rsfmri_volume
                                        )

                if control == 'mean':
                    target_age = _load_mean(dataset_sex)
                elif control == 'null':
                    target_age = np.array(0.0)
                elif control == 'both':
                    target_age_mean = _load_mean(dataset_sex)
                    target_age_null = np.array(0.0)
                    target_age = _load_target(
                        volume_index,
                        test_ages
                    )
                else:
                    target_age = _load_target(
                        volume_index,
                        test_ages
                    )
                if control == 'both':
                    age_delta, loss = _statistics_calculator(output_age, target_age)
                    age_delta_mean, loss_mean = _statistics_calculator(output_age, target_age_mean)
                    age_delta_null, loss_null = _statistics_calculator(output_age, target_age_null)
                    output_statistics[subject] = [target_age, output_age, age_delta, loss, age_delta_mean, loss_mean, age_delta_null, loss_null]
                else:
                    age_delta, loss = _statistics_calculator(output_age, target_age)
                    output_statistics[subject] = [target_age, output_age, age_delta, loss]


                log.info("Processed: " + volumes_to_be_used[volume_index] + " " + str(volume_index + 1) + " out of " + str(len(volumes_to_be_used)))

            except FileNotFoundError as exception_expression:
                log.error("Error in reading the provided file!")
                log.exception(exception_expression)
                if exit_on_error:
                    raise(exception_expression)

            except Exception as exception_expression:
                log.error("Error code execution!")
                log.exception(exception_expression)
                if exit_on_error:
                    raise(exception_expression)

        if control == 'both':
            columns=['target_age', 'output_age', 'age_delta', 'loss', 'age_delta_mean', 'loss_mean', 'age_delta_null', 'loss_null']
            output_statistics_df = pd.DataFrame.from_dict(output_statistics, orient='index', columns=columns)
        else:
            output_statistics_df = pd.DataFrame.from_dict(output_statistics, orient='index', columns=['target_age', 'output_age', 'age_delta', 'loss'])     
        output_statistics_df.to_csv(output_statistics_path)

    log.info("Output Data Generation Complete")

def _generate_age(file_path: str,
                model: nn.Module,
                device: str,
                cuda_available: bool,
                modality_flag: str,
                resolution: int,
                scale_factor: float,
                rsfmri_volume: int
                ) -> np.ndarray:
    
    """ Function wrapper for generating the age prediction

    Parameters:
    -----------
    file_path : str
        Path to the file
    model : nn.Module
        Model
    device : str
        Device
    cuda_available : bool
        Whether cuda is available or not
    modality_flag : str
        Modality flag
    resolution : int
        Resolution
    scale_factor : float
        Scale factor
    rsfmri_volume : int
        rsfmri volume

    Returns:
    --------
    output : np.ndarray
        Output

    """

    volume = data_utils.load_and_preprocess_evaluation(file_path=file_path, modality_flag=modality_flag, 
                                                        resolution=resolution, scale_factor=scale_factor, 
                                                        rsfmri_volume=rsfmri_volume)

    if len(volume.shape) != 5:
        volume = volume[np.newaxis, np.newaxis, :, :, :]

    volume = torch.tensor(volume).type(torch.FloatTensor)

    if cuda_available and (type(device) == int):
        volume = volume.cuda(device)

    output = model(volume)
    output = (output.cpu().numpy()).astype('float32')
    output = np.squeeze(output)

    return output


def _load_mean(dataset_sex: str) -> np.ndarray:
    """ Function wrapper for loading the mean age
    
    Parameters:
    -----------
    dataset_sex : str
        Sex of the subjects in the dataset under consideration

    Returns:
    --------
    mean_age : np.ndarray
        Mean age

    
    """

    if dataset_sex == 'male':
        mean_age = 63.370316719492536
    else:
        mean_age = 64.64810970818492

    mean_age = np.array(np.float32(mean_age))

    return mean_age 


def _load_target(volume_index: int, 
                 test_ages: str) -> np.ndarray:
    
    """ Function wrapper for loading the target age

    Parameters:
    -----------
    volume_index : int
        Volume index
    test_ages : str
        Path to the test ages
        
    Returns:
    --------
    target_age : np.ndarray
        Target age

    """

    target_age = np.load(test_ages)[volume_index]

    return target_age


def _statistics_calculator(output_age: np.ndarray, 
                           target_age: np.ndarray) -> tuple:

    """ Function wrapper for calculating the statistics

    Parameters:
    -----------
    output_age : np.ndarray
        Output age
    target_age : np.ndarray
        Target age

    Returns:
    --------    
    age_delta : np.ndarray
        Age delta
    loss : np.ndarray
        MSE loss

    """

    output_age = np.array(output_age)
    target_age = np.array(target_age)
    age_delta = output_age - target_age
    loss = MSELoss(torch.from_numpy(output_age), torch.from_numpy(target_age)).numpy()

    return age_delta, loss
