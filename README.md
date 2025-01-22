# Animal-Classifier-with-ResNet
PyTorch implementation of a neural network to classify images of animals using ResNet. The code was written for the [VLG Pixel Play Challenge](https://www.kaggle.com/competitions/vlg-recruitment-24-challenge/) and utitlises datasets provided in the contest for training.

# Features
1. **ResNet18:** the model is trained using the ResNet18 architecture, using its pretrained weights.
2. **Hyperparameters:** The model is currently set to train for 15 epochs, which was obtained after some tuning.

# Setup
1. Clone the repository.
2. Ensure you have all the Python packages installed. If not, run `pip install -r requirements.txt` in your virtual environment.
3. Download the [dataset](https://www.kaggle.com/competitions/vlg-recruitment-24-challenge/data).
4. (or) You may run this code on Kaggle itself.

# Run
Firstly, you will have to modify `train_dir` and `test_dir` in your code depending on its location.
Run the code using `python3 resnet.py`
