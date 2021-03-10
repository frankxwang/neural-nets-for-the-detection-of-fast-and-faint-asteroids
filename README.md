# The Fast and Inconspicuous: New Near Earth Asteroids Discovered in Zwicky Transient Facility Data Using Neural Networks and Artificial Data Generation are Both Fast and Faint

This repository contains the main scripts used for my research along with numerous Jupyter Notebooks used for experimentation and testing. 

## Main Scripts:
```generate_batches.py```: Generates the artificial dataset of asteroids used to train the machine learning model. 

```train.py```: Takes in that dataset and trains a Convolutional Neural Network, built upon EfficientNet-B1, to recognize asteroid streaks. The training results and trained models are recording using [Weights and Biases](https://wandb.ai).

```run_pipeline.py```: Uses the trained neural network to find asteroid streaks in a full night of data containing science, reference, and differenced images. 

```Process Full Data Night.ipynb```: Takes in the detections from ```run_pipeline.py```, applies further false positive reduction, and allows for the manual vetting and linking of asteroid streaks. The streaks are then checked to see if they belong to previously discovered ones and the visual magnitude and motion rates are measured. Currently, I am working on creating a better interface for this that does not require the use of a Jupyter Notebook.

## Extra Notebooks:
The Jupyter notebooks contained the ```Notebooks``` folder contain a selection of notebooks which I used for the testing and development of my research, highlighting the many different methods and algorithms I tried before settling on my current methodology. A brief description of what each notebook does is included at the top of each respective notebook.
