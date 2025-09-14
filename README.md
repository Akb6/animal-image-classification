# Animal Image Classification Project

Welcome to the Animal Image Classification project! This is a hands-on machine learning project designed to classify images of 15 different animal species using state-of-the-art transfer learning techniques. It leverages the powerful MobileNetV2 architecture pre-trained on ImageNet to achieve robust and accurate classification results even with a modest dataset.

---

## 🌟 Project Highlights

- **Deep Learning with Transfer Learning:** Uses MobileNetV2 as the feature extractor, significantly reducing training time and improving accuracy.
- **Data Augmentation:** Enhances model generalization by applying random image transformations during training.
- **Comprehensive Evaluation:** Prints detailed classification metrics — precision, recall, F1-score — for each animal class.
- **Easy to Use:** Clear instructions for setup and training with minimal dependencies.

---

## 🐾 Dataset Structure

Organize your dataset under a folder named `data/` in the root of this project. Each animal species should have its own directory named exactly after the class label. For example:

data/
├── Bear/
├── Bird/
├── Cat/
├── Cow/
├── Deer/
├── Dog/
├── Dolphin/
├── Elephant/
├── Giraffe/
├── Horse/
├── Kangaroo/
├── Lion/
├── Panda/
├── Tiger/
└── Zebra/

Each folder should contain relevant images of that animal.

---

## ⚙️ Setup Guide

### 1. Install Python (Version 3.7 - 3.11 recommended)

Make sure Python is installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Install Required Packages

Since `requirements.txt` is not included, install the dependencies manually. Open your terminal or command prompt and run:

pip install tensorflow numpy matplotlib scikit-learn


### 3. Prepare Your Dataset

Place your image folders inside the `data/` directory as described above.

### 4. Run the Training Script

In your terminal, navigate to the project directory and execute:

python animal_classifier.py


The model will train over 10 epochs, showing training and validation accuracy and loss in real-time.

---

## 📊 What to Expect

- **Training Progress:** Real-time logging of accuracy and loss on training and validation sets.
- **Evaluation Report:** After training, a detailed classification report displays performance metrics for each class, helping you identify strengths and areas for improvement.
- **Model Save:** The trained model is saved as `animal_classifier_model.h5` for easy reuse or deployment.

---

## 💡 Tips & Tricks

- **Fine-Tuning:** For better performance, consider unfreezing some of the base MobileNetV2 layers and training for more epochs.
- **Data Quality:** Ensure your dataset is balanced and images are clear for optimal results.
- **Experimentation:** Try augmenting the data further or tweaking model parameters to boost accuracy.

---

## 🚀 Next Steps

- Deploy your trained model as a web or mobile app.
- Expand the dataset with more animal classes.
- Incorporate advanced architectures or hyperparameter tuning.


Thank you for exploring this project! If you have questions or want to contribute, feel free to open an issue or pull request on GitHub.

Happy coding and happy classifying! 🐅🐘🦒
