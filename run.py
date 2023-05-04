import os
import shutil
import argparse
import logging

import torch
import torch.utils.data as data
import numpy as np

from solver import Solver
from AgeMapper import AgeMapper
from utils.data_utils import get_datasets, get_datasets_dynamically
from utils.settings import Settings
import utils.data_evaluation as evaluations
from utils.misc import my_KLDivLoss


# Set the default floating point tensor type to FloatTensor
torch.set_default_tensor_type(torch.FloatTensor)


def load_data(data_parameters: dict) -> tuple:
    """
    Function to load the training and validation datasets

    Parameters
    ----------
    data_parameters : dict
        Dictionary containing the data parameters

    Returns
    -------
    tuple
        Tuple containing the training and validation datasets

    """

    print("Data is loading...")
    train_data, validation_data = get_datasets(data_parameters)
    print("Data has loaded!")
    print("Training dataset size is {}".format(len(train_data)))
    print("Validation dataset size is {}".format(len(validation_data)))

    return train_data, validation_data

def load_data_dynamically(data_parameters: dict) -> tuple:
    """
    Function to load the training and validation datasets in a dynamic fashion.
    This means that rather than pre-processing the data before training, we do it on the fly.
    This is useful when we have a large dataset and we want to save space on the disk.

    Parameters
    ----------
    data_parameters : dict
        Dictionary containing the data parameters

    Returns
    -------
    tuple
        Tuple containing the training and validation datasets
    
    """
    print("Data is loading...")
    train_data, validation_data, resolution = get_datasets_dynamically(data_parameters)
    print("Data has loaded!")
    print("Training dataset size is {}".format(len(train_data)))
    print("Validation dataset size is {}".format(len(validation_data)))

    return train_data, validation_data, resolution


def train(data_parameters: dict, 
          training_parameters: dict, 
          network_parameters: dict, 
          misc_parameters: dict) -> None:
    
    """
    Function to train the AgeMapper network

    Parameters
    ----------
    data_parameters : dict
        Dictionary containing the data parameters
    training_parameters : dict
        Dictionary containing the training parameters
    network_parameters : dict
        Dictionary containing the network parameters
    misc_parameters : dict
        Dictionary containing the misc parameters

    Returns
    -------
    None

    """

    # Set the optimiser and optimiser arguments

    if training_parameters['optimiser'] == 'adamW':
        optimizer = torch.optim.AdamW
    elif training_parameters['optimiser'] == 'adam':
        optimizer = torch.optim.Adam
    else:
        optimizer = torch.optim.Adam # Default option

    optimizer_arguments={'lr': training_parameters['learning_rate'],
                        'betas': training_parameters['optimizer_beta'],
                        'eps': training_parameters['optimizer_epsilon'],
                        'weight_decay': training_parameters['optimizer_weigth_decay']
                        }
    
    # Set the loss function

    if training_parameters['loss_function'] == 'mse':
        loss_function = torch.nn.MSELoss()
    elif training_parameters['loss_function'] == 'mae':
        loss_function = torch.nn.L1Loss()
    else:
        print("Loss function not valid. Defaulting to MSE!")
        loss_function = torch.nn.MSELoss(reduction='batchmean')

    # Define the training and validation data loaders. 
    # If pre_processing_before_training is set to True, the data will be pre-processed before training.
    # If pre_processing_before_training is set to False, the data will be pre-processed on the fly during training. This is the default option.

    if data_parameters['pre_processing_before_training'] == True:
        train_data, validation_data = load_data(data_parameters)
        train_loader = data.DataLoader(
            dataset=train_data,
            batch_size=training_parameters['training_batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )
        validation_loader = data.DataLoader(
            dataset=validation_data,
            batch_size=training_parameters['validation_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )
    else:
        train_data, validation_data, resolution = load_data_dynamically(data_parameters)
        train_loader = data.DataLoader(
            dataset=train_data,
            batch_size=training_parameters['training_batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )
        validation_loader = data.DataLoader(
            dataset=validation_data,
            batch_size=training_parameters['validation_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )

    # Define the dropout rates and the model

    dropout_rate_1 = network_parameters['dropout_rate_1']
    dropout_rate_2 = network_parameters['dropout_rate_2']
    dropout_rate_3 = network_parameters['dropout_rate_3']

    # Load the model or create a new one

    if training_parameters['use_pre_trained']:
        pre_trained_path = "saved_models/" + training_parameters['experiment_name'] + ".pth.tar"
        AgeMapperModel = torch.load(pre_trained_path)
    else:
        AgeMapperModel = AgeMapper(resolution=resolution,
                                    dropout_rate_1=dropout_rate_1,
                                    dropout_rate_2=dropout_rate_2,
                                    dropout_rate_3=dropout_rate_3,
                                    )
        
    # Define the solver and train the model

    solver = Solver(model=AgeMapperModel,
                    number_of_classes=network_parameters['number_of_classes'],
                    experiment_name=training_parameters['experiment_name'],
                    optimizer=optimizer,
                    optimizer_arguments=optimizer_arguments,
                    loss_function=loss_function,
                    model_name=training_parameters['experiment_name'],
                    number_epochs=training_parameters['number_of_epochs'],
                    loss_log_period=training_parameters['loss_log_period'],
                    learning_rate_scheduler_step_size=training_parameters['learning_rate_scheduler_step_size'],
                    learning_rate_scheduler_gamma=training_parameters['learning_rate_scheduler_gamma'],
                    use_last_checkpoint=training_parameters['use_last_checkpoint'],
                    experiment_directory=misc_parameters['experiments_directory'],
                    logs_directory=misc_parameters['logs_directory'],
                    checkpoint_directory=misc_parameters['checkpoint_directory'],
                    best_checkpoint_directory=misc_parameters['best_checkpoint_directory'],
                    save_model_directory=misc_parameters['save_model_directory'],
                    learning_rate_validation_scheduler=training_parameters['learning_rate_validation_scheduler'],
                    learning_rate_cyclical = training_parameters['learning_rate_cyclical'],
                    learning_rate_scheduler_patience=training_parameters['learning_rate_scheduler_patience'],
                    learning_rate_scheduler_threshold=training_parameters['learning_rate_scheduler_threshold'],
                    learning_rate_scheduler_min_value=training_parameters['learning_rate_scheduler_min_value'],
                    learning_rate_scheduler_max_value=training_parameters['learning_rate_scheduler_max_value'],
                    learning_rate_scheduler_step_number=training_parameters['learning_rate_scheduler_step_number'],
                    early_stopping_patience=training_parameters['early_stopping_patience'],
                    early_stopping_min_patience=training_parameters['early_stopping_min_patience'],
                    early_stopping_min_delta=training_parameters['early_stopping_min_delta'],
                    )

    solver.train(train_loader, validation_loader)

    # Free up memory

    del train_data, validation_data, train_loader, validation_loader, AgeMapperModel, solver, optimizer
    torch.cuda.empty_cache()


def evaluate_data(mapping_evaluation_parameters: dict, 
                  data_parameters: dict, 
                  ) -> None:
    
    """
    Function to evaluate the AgeMapper network on the test dataset

    Parameters
    ----------
    mapping_evaluation_parameters : dict
        Dictionary containing the mapping evaluation parameters
    data_parameters : dict
        Dictionary containing the data parameters

    Returns
    -------
    None

    """

    experiment_name = mapping_evaluation_parameters['experiment_name']

    trained_model_path = "saved_models/" + experiment_name + ".pth.tar"
    prediction_output_path = experiment_name + "_predictions"

    data_directory = mapping_evaluation_parameters['data_directory']
    
    # dataset_sex = mapping_evaluation_parameters['dataset_sex']----- WE WILL ADD THIS LATER, BUT NOW USE >>JOB.INI<<< FILE FOR MODALITY. EASIER!
    # dataset_size = mapping_evaluation_parameters['dataset_size']----- WE WILL ADD THIS LATER, BUT NOW USE >>JOB.INI<<< FILE FOR MODALITY. EASIER!
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']

    prediction_output_statistics_name = mapping_evaluation_parameters['prediction_output_statistics_name']

    if mapping_evaluation_parameters['dataset_type'] == 'validation':
        data_list = mapping_evaluation_parameters['male_validation']
        test_ages = mapping_evaluation_parameters['male_validation_age']
        prediction_output_statistics_name += '_validation.csv'
    elif mapping_evaluation_parameters['dataset_type'] == 'train':
        data_list = data_parameters['male_train']
        test_ages = data_parameters['male_train_age']
        prediction_output_statistics_name += '_train.csv'
    else:
        data_list = mapping_evaluation_parameters['male_test']
        test_ages = mapping_evaluation_parameters['male_test_age']
        prediction_output_statistics_name += '_test.csv'

    if dataset_sex == 'female':
        data_list = "fe" + data_list
        test_ages = "fe" + test_ages

    if dataset_size == 'small':
        data_list += "_small.txt"
        test_ages += "_small.npy"
    elif dataset_size == 'tiny':
        data_list += "_tiny.txt"
        test_ages += "_tiny.npy"
    elif 'small' in dataset_size and dataset_size!='small':
        data_list += "_small.txt"
        test_ages += "_small.npy"
    else:
        data_list += ".txt"
        test_ages += ".npy"

    if dataset_size == 'han':
        data_list = "test_han.txt"
        test_ages = "test_age_han.npy"

    if dataset_size == 'everything':
        data_list = "test_everything.txt"
        test_ages = "test_age_everything.npy"

    data_list = 'datasets/' + data_list
    test_ages = 'datasets/' + test_ages

    # modality_flag = mapping_evaluation_parameters['modality_flag'] ----- WE WILL ADD THIS LATER, BUT NOW USE >>JOB.INI<<< FILE FOR MODALITY. EASIER!
    modality_flag = data_parameters['modality_flag'] 

    control = mapping_evaluation_parameters['control']
    device = mapping_evaluation_parameters['device']
    exit_on_error = mapping_evaluation_parameters['exit_on_error']

    scaling_values = mapping_evaluation_parameters['scaling_values']

    evaluations.evaluate_data(trained_model_path,
                     data_directory,
                     data_list,
                     prediction_output_path,
                     prediction_output_statistics_name,
                     modality_flag,
                     control,
                     device,
                     test_ages,
                     dataset_sex,
                     scaling_values,
                     exit_on_error
                    )


def delete_files(folder: str) -> None:
    """
    Function to delete all files in a folder

    Parameters
    ----------
    folder : str
        Folder to delete all files from

    Returns
    -------
    None
        
    """
    for object_name in os.listdir(folder):
        file_path = os.path.join(folder, object_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            print(exception)


if __name__ == '__main__':

    # Main script for running the training and evaluation of the AgeMapper network
    # The script takes in a mode argument which can be either train, evaluate-data, evaluate-mapping, train-and-evaluate-mapping, clear-checkpoints, clear-logs, clear-experiment and clear-everything (req uncomment for safety!)
    # The script also takes in a model_name argument which is used to identify the settings file modelName.ini & modelName_eval.ini
    # The script also takes in a use_last_checkpoint argument which is used to identify if the last checkpoint should be used if 1; useful when wanting to time-limit jobs.
    # The script also takes in a number_of_epochs argument which is used to identify how many epochs the network will train for; should be limited to ~3 hours or 2/3 epochs

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True,
                        help='run mode, valid values are train, evaluate-data, clear-checkpoints, clear-checkpoints-completely, clear-logs, clear-experiment, clear-experiment-completely, train-and-evaluate-mapping, lr-range-test, solver-logger-test')
    parser.add_argument('--model_name', '-n', required=True,
                        help='model name, required for identifying the settings file modelName.ini & modelName_eval.ini')
    parser.add_argument('--use_last_checkpoint', '-c', required=False,
                        help='flag indicating if the last checkpoint should be used if 1; useful when wanting to time-limit jobs.')
    parser.add_argument('--number_of_epochs', '-e', required=False,
                        help='flag indicating how many epochs the network will train for; should be limited to ~3 hours or 2/3 epochs')

    arguments = parser.parse_args()

    settings_file_name = arguments.model_name + '.ini'
    evaluation_settings_file_name = arguments.model_name + '_eval.ini'

    settings = Settings(settings_file_name)
    data_parameters = settings['DATA']
    training_parameters = settings['TRAINING']
    network_parameters = settings['NETWORK']
    misc_parameters = settings['MISC']

    if arguments.use_last_checkpoint == '1':
        training_parameters['use_last_checkpoint'] = True
    elif arguments.use_last_checkpoint == '0':
        training_parameters['use_last_checkpoint'] = False

    if arguments.number_of_epochs is not None:
        training_parameters['number_of_epochs'] = int(arguments.number_of_epochs)

    if arguments.mode == 'train':
        train(data_parameters, training_parameters, network_parameters, misc_parameters)

    elif arguments.mode == 'evaluate-data':
        logging.basicConfig(filename='evaluate-data-error.log')
        settings_evaluation = Settings(evaluation_settings_file_name)
        mapping_evaluation_parameters = settings_evaluation['MAPPING']
        evaluate_data(mapping_evaluation_parameters, data_parameters, network_parameters)

    elif arguments.mode == 'clear-checkpoints':

        warning_message = input("Warning! This command will delete all checkpoints. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                print('Cleared the current experiment checkpoints successfully!')
            else:
                print('ERROR: Could not find the experiment checkpoints.')
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-checkpoints-completely':
        warning_message = input("WARNING! This command will delete all checkpoints (INCL BEST). DANGER! Continue [y]/n: ")
        if warning_message == 'y':
            warning_message2 = input("ARE YOU SURE? [y]/n: ")
            if warning_message2 == 'y':
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                    print('Cleared the current experiment checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory']))
                    print('Cleared the current experiment best checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment best checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment folder successfully!')
                else:
                    print("ERROR: Could not find the experiment folder.")
            else:
                print("Action Cancelled!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-logs':

        warning_message = input("Warning! This command will delete all checkpoints and logs. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                print('Cleared the current experiment logs directory successfully!')
            else:
                print("ERROR: Could not find the experiment logs directory!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-experiment':

        warning_message = input("Warning! This command will delete all checkpoints and logs. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                print('Cleared the current experiment checkpoints successfully!')
            else:
                print('ERROR: Could not find the experiment checkpoints.')
            if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                print('Cleared the current experiment logs directory successfully!')
            else:
                print("ERROR: Could not find the experiment logs directory!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-experiment-completely':
        warning_message = input("WARNING! This command will delete all checkpoints (INCL BEST) and logs. DANGER! Continue [y]/n: ")
        if warning_message == 'y':
            warning_message2 = input("ARE YOU SURE? [y]/n: ")
            if warning_message2 == 'y':
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                    print('Cleared the current experiment checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory']))
                    print('Cleared the current experiment best checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment best checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment folder successfully!')
                else:
                    print("ERROR: Could not find the experiment folder.")
                if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment logs directory successfully!')
                else:
                    print("ERROR: Could not find the experiment logs directory!")
            else:
                print("Action Cancelled!")
        else:
            print("Action Cancelled!")

    # elif arguments.mode == 'clear-everything':
    #     delete_files(misc_parameters['experiments_directory'])
    #     delete_files(misc_parameters['logs_directory'])
    #     print('Cleared the all the checkpoints and logs directory successfully!')

    elif arguments.mode == 'train-and-evaluate-data':
        settings_evaluation = Settings(evaluation_settings_file_name)
        mapping_evaluation_parameters = settings_evaluation['MAPPING']
        train(data_parameters, training_parameters,
              network_parameters, misc_parameters)
        logging.basicConfig(filename='evaluate-mapping-error.log')
        evaluate_data(mapping_evaluation_parameters)
      
    else:
        raise ValueError('Invalid mode value! Only supports: train, evaluate-data, evaluate-mapping, train-and-evaluate-mapping, clear-checkpoints, clear-logs,  clear-experiment and clear-everything (req uncomment for safety!)')
