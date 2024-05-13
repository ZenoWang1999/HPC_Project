# Sentimental Analysis of IMDB 50k dataset with GPU-based Parallel Computing Optimized LSTM

## Project Description

This project aims to develop a sentiment analysis model for movie reviews using LSTM networks. The input dataset consists of 50,000 movie reviews from the IMDB, along with their corresponding sentiment labels, either positive or negative.

The project will establish a baseline LSTM model for sentiment analysis and then deploy it to a HPC environment. To improve the model's training speed in the distributed environment, data prefetching and mixed-precision models will be implemented.

The project will focus on two key metrics: the training speedup of the models using various optimization techniques and the change in model accuracy relative to the baseline. The ultimate goal is to significantly improve the model's training speed while minimizing the decline in accuracy.

By leveraging advanced techniques in an HPC environment, this project aims to enhance the efficiency of sentiment analysis for movie reviews while maintaining high performance standards.

## Milestones and Completion Status
- [x] __Literature Review and Model Selection:__ Review relevant literature on sentiment analysis and LSTM models
- [x] __Dataset Collection and Preprocessing:__ Acquire and preprocess the IMDB movie review dataset.
- [x] __Model selection and Training:__ Choose the proper models for the task, and do the training process with the IMDB dataset.
- [x] __Model Accuracy Optimization:__ Pick up the best model that optimal the final result
- [x] __GPU-accelerated Training Optimization:__ Implement GPU-accelerated training with data prefetching and mixed-precision models.
- [ ] __Inference Speed Optimization:__ Implement inference speed optimization with knowledge distillation and quantization.
- [x] __Evaluation and Analysis__: Compare and analyze the training speed and accuracy of the original model with the model using data prefetching and mixed precision.
- [x] __Report Writing__: Write a detailed report on the project, including the methodology, results, and conclusions.

__In the original plan, we intended to implement knowledge distillation in the model inference part to accelerate the inference process. However, in subsequent practice, we found that due to the small input data dimension and the small number of layers in the original model, implementing knowledge distillation did not significantly speed up the inference process and instead severely affected the model accuracy.__

## Repository Structure
The following is a description of each file:

1. `README.md`: Provides an overview of the project, milestones, and instructions for use.
2. `Midpoint Project Checkpoint.pdf`: The project's midterm report, serving as a reference for the project setup process.
3. `IMDB Dataset.csv`: The dataset used for sentiment analysis, containing 50,000 movie reviews and their corresponding sentiment labels.
4. `preprocessed_data.csv`: The dataset of the original data after preprocessing, such as word segmentation.
5. `preprocess.py`: The script for preprocessing the original dataset. Data preprocessing is only used for local execution and not for HPC execution. If you want to train the model, please directly use `train_optimized.sbatch`.
6. `mixed_precision.py`: The predefined function for training the model with mixed precision.
7. `original_model.py`: The script for training the original model without any optimization.
8. `train.sbatch`: The script for training the original model on the HPC environment.
9. `optimized_model.py`: The script for training the optimized model with data prefetching and mixed-precision models.
10. `train_optimized.sbatch`: The script for training the optimized model on the HPC environment.
11. `result` folder: The results of experiments, including the training speed and accuracy of the original and optimized models.
    

## Instructions for Use
1. Clone the repository to your HPC.
2. Use the command `virtualenv -p python $SCRATCH/myProject` to create a virtual environment named `myProject` in the `scratch` folder.
3. Use `source $SCRATCH/myProject/bin/activate` to activate the virtual environment, and use `pip install tensorflow` to install TensorFlow.
4. Use `sbatch train_optimized.sbatch` to run the script. The results will be output to `optimized.out`.

## Results
The optimized model shows 1.55x speed up in average training timen per epoch compared to the original model.
![avg_time_epoch](https://github.com/ZenoWang1999/HPC_Project/blob/master/results/avg_time_epoch.png)
The optimized model shows a slight decrease in accuracy compared to the original model. But the difference is not significant.
![acc_train_acc](https://github.com/ZenoWang1999/HPC_Project/blob/master/results/avg_train_acc.png)
![acc_val_acc](https://github.com/ZenoWang1999/HPC_Project/blob/master/results/avg_vali_acc.png)
The optimized model shows a 1.44 speedup in data loading time compared to the original model.
![dataload_time](https://github.com/ZenoWang1999/HPC_Project/blob/master/results/dataload_time.png)

