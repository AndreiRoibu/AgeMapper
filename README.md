# AgeMapper

This is a working release of AgeMapper, also reffered to as HGL. For any issues, please contact Andrei Roibu at andrei-claudiu.roibu@dtc.ox.ac.uk. 

The code is still undergoing modifications and further additions will be made in the coming period.

The code is also released together with the following materials. **To access these materials, please see more information in the sections below** 

* Pre-trained models for each of the 57 contrasts utilised in this work. For each contrast, and for both male and female subjects, 3 identical networks were trained. To reduce stochastic noise in predictions, it is advised that at inference the predictions are averaged at the subject level.
* A dataset containing the associations between non-imaging phenotypes (nIDPs) from the UK Biobank, and the brain age deltas obtained with the networks.
* The subject datasets used for training, together with a file containing information about each contrast.

## Motivation

Brain ageing is a highly variable, spatially and temporally heterogeneous process, marked by numerous structural and functional changes. These can cause discrepancies between individuals’ chronological age and the apparent age of their brain, as inferred from neuroimaging data. Machine learning models, and particularly Convolutional Neural Networks (CNNs), have proven adept in capturing patterns relating to ageing induced changes in the brain. The differences between the predicted and chronological ages, or brain age deltas, have emerged as useful biomarkers for exploring those factors which promote accelerated ageing or resilience, such as pathologies or lifestyle factors. However, studies rely only on structural neuroimaging for predictions, overlooking potentially informative functional and microstructural changes. Here we show that multiple contrasts derived from different MRI modalities can predict brain age, each encoding bespoke brain ageing information. By using 3D CNNs and UK Biobank data, we found that 57 contrasts derived from structural, susceptibility-weighted, diffusion, and functional MRI can successfully predict brain age. For each contrast, different patterns of association with nonimaging phenotypes were found, resulting in a total of 191 unique, statistically significant associations. Furthermore, we found that ensembling data from multiple contrasts results in both higher prediction accuracies and stronger correlations to non-imaging measurements. Our results demonstrate that other 3D contrasts and modalities, which have not been considered so far for the task of brain age prediction, encode different information about the ageing brain. We envision our work as being the starting point for future investigations into the causal links underpinning the observed brain age deltas and non-imaging measurement associations. For instance, drug effects can be monitored, given that certain medications correlate with accelerated brain ageing. Furthermore, continued development of brain age models could facilitate their deployment in clinical trials for recruitment and monitoring, and hospitals for diagnostic and screening tasks.

## Network Architecture & Pre-Trained Networks

The network architecture is presented in the following figure. The next figure gives an indication on how the brain ages are calculated by ensembling three identical networks. To access the pre-trained networks, please use [THIS LINK](). For information on the specific code names for each contrast and modality, please look [HERE]().  **ATTENTION - This data has not yet been added and the links do not work. Updates to follow** 

![network architecture](/figures/HGL.png)

![network architecture](/figures/HGLensemble.png)


## Subject Datasets and Contrast Information

To access the datasets utilised for training these networks, see the various text and numpy files in the __datasets__ folder in this repository. The actual MRI scans are available upon application from the [UK Biobank](https://www.ukbiobank.ac.uk), such as all the other data utilised in this project. 

In the __datasets__ there is also a file named __scaling_values_simple.csv__. This CSV file contains information on the name of each of the contrasts, the scale factor utilised during data pre-processing, the resolution of the MRI files, and the internal UK Biobank file handle marking where the file can be found for each subject.

## Correlations to UK Biobank nIDPs

One of the major findings of this work has been that all contrasts correlate significantly and differently with a large number of nIDPs from the UK Biobank. The correlations and the information for accessing the data is made freely available for the research community to further investigate. All the correlations can be accessed at [THIS LINK](). **These are very large files!**. For smaller files, containing only the statistically significant associations, please use [THIS LINK](), or for an interactive visualisation of the data, [THIS LINK](). **ATTENTION - This data has not yet been added and the links do not work. Updates to follow**  

## Installation & Usage
To download and run this code as python files, you can clone this repository using git:

```bash
git clone <link to repo>
```

In order to install the required packages, the user will need the presence of Python3 and the [pip3](https://pip.pypa.io/en/stable/) installer. 

For installation on Linux or OSX, use the following commands. This will create a virtual environment and automatically install all the requirements, as well as create the required metadata

```bash
./setup.sh
```

In order to run the code, activate the previously installed virtual environment, and utilise the run file. Several steps are needed prior to that:
* make a copy of the __settings.ini__ and __settings_eval.ini__ files, filling out the required settings. If running an evaluation, make sure that the pre-trained network name corresponds to the experiment names
* rename the two __ini__ files to either the pre-trained network name, or to something else

This approach has been used given the large number of hyperparameters and network-subject sex-MRI modality combinations.

After setting up, activate your environment using the following code:

```bash
~/(usr)$ source env/bin/activate
```

For running network training epochs, utilise this code, setting TASK='train' (or test), NAME='name_of_your_ini_files', CHECKPOINT='0' (or some other value if wishing to start from a later checkpoint), and EPOCHS='number_of_epochs_you_want_to_train_for'. For more details on these inputs, see the __run.py__ file.

```bash
~/(usr)$ python run.py -m ${TASK} -n ${NAME} -c ${CHECKPOINT} -e ${EPOCHS}
```


## References

The work published in this repository has been published at the 2023 10th Swiss Conferece on Data Science (SDS). The paper can be accessed at [THIS LINK](). To reference this work, please use the following citation.

```
@inproceedings{roibu2023brain,
  title={Brain Ages Derived from Different MRI Modalities are Associated with Distinct Biological Phenotypes},
  author={Roibu, Andrei and Adaszewski, Stanislaw and Schindler, Torsten and Smith, Stephen and Namburete, Ana and Lange, Frederik},
  booktitle={2023 10th Swiss Conference on Data Science (SDS)},
  pages={1--9},
  year={2023},
  organization={IEEE}
}
```

In the creation of this code, material was used from the following paper. The original code can be accessed on [GitHub](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/)

```
@article{peng2021accurate,
  title={Accurate brain age prediction with lightweight deep neural networks},
  author={Peng, Han and Gong, Weikang and Beckmann, Christian F and Vedaldi, Andrea and Smith, Stephen M},
  journal={Medical image analysis},
  volume={68},
  pages={101871},
  year={2021},
  publisher={Elsevier}
}
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Licence
[BSD 3-Clause Licence](https://opensource.org/licenses/BSD-3-Clause) © [Andrei Roibu](https://github.com/AndreiRoibu)
