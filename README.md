# PyTorch Learning Repository — Full Technical Documentation

This document provides a detailed explanation of every major step involved in setting up the environment, installing PyTorch, training a neural network, handling datasets, saving/loading models, structuring the repository, and managing large model files using Git LFS. It is written for clarity and completeness.

------------------------------------------------------------
1. Conda Environment Setup
------------------------------------------------------------

Conda is a package and environment manager used to create isolated workspaces. Each environment can have its own Python version and dependencies.

Commands used:
conda create -n torch-env python=3.11 -y
conda activate torch-env

Purpose:
- Keeps machine learning projects isolated
- Prevents package conflicts
- Allows reproducibility

Terminology:
- Environment: A self-contained Python workspace
- Dependency: External library needed by a project

------------------------------------------------------------
2. Installing PyTorch with MPS Acceleration (Apple Silicon)
------------------------------------------------------------

PyTorch supports MPS (Metal Performance Shaders), Apple’s GPU acceleration backend for deep learning on M1/M2/M3 devices.

Command used:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Notes:
- Despite the "cpu" label, this wheel includes MPS support.
- MPS drastically accelerates matrix operations and model training.

Terminology:
- MPS: Metal Performance Shaders
- GPU: Graphics Processing Unit, used for parallel computation
- Wheel: A Python package distribution format

------------------------------------------------------------
3. Datasets and DataLoader
------------------------------------------------------------

PyTorch provides pre-built datasets and tools for loading them efficiently.

Dataset used:
FashionMNIST

Key components:
- Dataset: Contains images and labels
- DataLoader: Converts the dataset into batches, shuffles data, and loads it efficiently

Purpose:
- Batching improves training speed
- Shuffling improves model generalization
- DataLoader abstracts low-level loading details

Terminology:
- Batch: A group of samples processed at once
- Shuffle: Random reordering of samples to prevent learning order patterns

------------------------------------------------------------
4. Building a Neural Network Model
------------------------------------------------------------

The model used is a Multi-Layer Perceptron (MLP), defined using PyTorch’s nn.Module.

Layers:
- Flatten: Converts each image from 28x28 to a 784-dimensional vector
- Linear layers: Fully connected layers
- ReLU: Activation function introducing non-linearity

Terminology:
- MLP: Multi-Layer Perceptron
- Activation Function: Adds non-linearity to help the model learn complex patterns
- Forward Pass: The process of computing model output from input

------------------------------------------------------------
5. Training Loop Explained
------------------------------------------------------------

A training loop consists of the following steps:

1. Forward pass:
   pred = model(X)
   The model predicts outputs from the input batch.

2. Loss computation:
   loss = loss_fn(pred, y)
   Measures how wrong the model’s predictions are.

3. Backward pass (backpropagation):
   loss.backward()
   Calculates gradients for all learnable parameters.

4. Optimization step:
   optimizer.step()
   Updates model weights based on gradient values.

5. Reset gradients:
   optimizer.zero_grad()
   Prevents accumulation of gradients between training steps.

Terminology:
- Loss: Numerical measure of model error
- Gradient: Direction and magnitude of parameter updates
- Backpropagation: Algorithm for computing gradients
- SGD: Stochastic Gradient Descent, an optimization method

------------------------------------------------------------
6. Testing and Validation Loop
------------------------------------------------------------

Testing is performed without gradient calculations to improve speed.

Steps:
- Set model to evaluation mode using model.eval()
- Disable gradient computation using torch.no_grad()
- Compute loss and accuracy over the test dataset

Purpose:
- Measures generalization
- Detects overfitting or underfitting

Terminology:
- Evaluation Mode: Disables certain behaviors like dropout
- Accuracy: Percentage of correct predictions
- Generalization: Model performance on unseen data

------------------------------------------------------------
7. Saving and Loading Models
------------------------------------------------------------

PyTorch models are saved using state_dict, which maps each layer to its parameters.

Saving:
torch.save(model.state_dict(), "model.pth")

Loading:
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth", weights_only=True"))
model.eval()

Purpose:
- Allows training to be resumed
- Enables deployment and inference
- Ensures reproducibility

Terminology:
- state_dict: A dictionary containing model parameters
- Checkpoint: Saved snapshot of model weights

------------------------------------------------------------
8. Structuring a Machine Learning Repository
------------------------------------------------------------

A professional ML repository includes:

scripts/        Training and inference scripts
models/         Saved model files (tracked with Git LFS)
notes/          Documentation and learning notes
notebooks/      Jupyter notebooks (optional)
data/           Downloaded datasets (ignored by git)
README.md       Main documentation
.gitignore      Defines ignored files

Purpose:
- Makes the project modular and maintainable
- Improves collaboration and clarity
- Aligns with industry practices

Terminology:
- Repository: A version-controlled project folder
- Directory Structure: Organized layout of files and folders

------------------------------------------------------------
9. Git LFS (Large File Storage)
------------------------------------------------------------

Git LFS is required because model files (.pth) can be large and GitHub limits normal files to 100 MB.

Git LFS workflow:

Install:
brew install git-lfs
git lfs install

Track model files:
git lfs track "*.pth"
git add .gitattributes
git commit -m "Configure Git LFS"

Add model file:
mv ~/model.pth models/model.pth
git add -f models/model.pth
git commit -m "Add trained model via Git LFS"
git push

Purpose:
- Prevents bloating the repository
- Allows versioning of large files
- Ensures efficient uploads and downloads

Terminology:
- LFS: Large File Storage
- Pointer File: A lightweight reference replacing the actual large file
- Bandwidth Quota: GitHub LFS usage limits

------------------------------------------------------------
10. GitHub Repository Setup and Deployment
------------------------------------------------------------

Steps followed:

1. Created a new GitHub repository.
2. Cloned the repository locally:
   git clone https://github.com/<username>/pytorch-learning.git

3. Navigated into the project:
   cd pytorch-learning

4. Created project structure using mkdir commands.
5. Added and committed files using:
   git add .
   git commit -m "Initial commit"
   git push

6. Configured Git LFS for model storage.
7. Added trained model file using force addition.
8. Confirmed repository structure and documentation.

Purpose:
- Provides remote backup
- Enables collaboration
- Offers a version history of work

Terminology:
- Clone: Copy a remote Git repo locally
- Commit: Save changes to the repo history
- Push: Upload local commits to GitHub
- Remote: The cloud version of the repository

============================================================
11. Store Trained Models in GitHub Using Git LFS
============================================================

Step 1: Install Git LFS (one time)
----------------------------------
brew install git-lfs
git lfs install

Step 2: Tell Git LFS to track all .pth model files
--------------------------------------------------
git lfs track "*.pth"
git add .gitattributes
git commit -m "Configure Git LFS for model files"
git push origin main

Step 3: Move your trained model into the models folder
-------------------------------------------------------
mv ~/model.pth models/model.pth

Step 4: Force-add the model (because /models is in .gitignore)
--------------------------------------------------------------
git add -f models/model.pth
git commit -m "Add trained model via Git LFS"
git push origin main

Your model is now safely stored on GitHub using Git LFS.


============================================================
12. Run Your Training Script From the Terminal
============================================================

Step 1: Activate the conda environment
--------------------------------------
conda activate torch-env

Step 2: Run the training script
-------------------------------
python3 scripts/fashion_mnist_train.py --epochs 5 --batch-size 64

The script will:
- download FashionMNIST
- train the model
- save checkpoints to:   models/checkpoint_epochN.pth
- save final model to:   models/model_final.pth


------------------------------------------------------------
Summary of Key Full Forms and Concepts
------------------------------------------------------------

MLP   - Multi-Layer Perceptron  
ReLU  - Rectified Linear Unit  
MPS   - Metal Performance Shaders  
SGD   - Stochastic Gradient Descent  
LFS   - Large File Storage  
CPU   - Central Processing Unit  
GPU   - Graphics Processing Unit  
Epoch - One full pass through the dataset  
Batch - Subset of samples processed together  
Loss  - Error measure guiding model improvement  
