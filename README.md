# skin-cancer-detection
This project explores skin cancer detection using Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and Recurrent Neural Networks (RNN) for classifying benign and malignant lesions.

# Skin Cancer Detection: A Comparative Analysis of LSTM, RNN, and CNN

## Introduction
This project focuses on leveraging deep learning techniques for the early and accurate detection of skin cancer, a significant public health concern where timely diagnosis is crucial for effective treatment. We explore and compare the effectiveness of Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Long Short-Term Memory (LSTM) networks in classifying skin lesions as either benign or malignant. The aim is to contribute to the development of automated diagnostic tools that can assist clinicians in making more informed decisions and ultimately enhance patient care.

## Models Explored

### Convolutional Neural Networks (CNN)
A CNN model was constructed with multiple convolutional and max-pooling layers for hierarchical feature learning. The architecture includes convolutional layers with ReLU activation for spatial feature extraction, max-pooling layers for dimensionality reduction, a flatten layer, and fully connected (dense) layers with a final softmax activation for binary classification.

* **CNN Model Summary:**
    ![CNN Model Summary](path/to/cnn_model_summary.png)
    *Figure 2.2. CNN Model Summary*

### Recurrent Neural Networks (RNN)
The Simple RNN model was implemented for skin cancer detection, leveraging its ability to process sequential data. While computationally efficient, it showed limitations in capturing complex spatial features inherent in image data compared to CNNs or LSTM-based approaches.

* **RNN Model Summary:**
    ![RNN Model Summary](path/to/rnn_model_summary.png)
    *Figure 2.3. RNN Model Summary*

### Long Short-Term Memory (LSTM)
LSTM layers were integrated into the model to process sequential feature maps and extract temporal patterns, enhancing the model's ability to understand sequential complexities. The LSTM architecture includes `tanh` activation for cell state updates and `sigmoid` for gate mechanisms, along with dropout layers to mitigate overfitting. This approach combines ResNet50's feature extraction with LSTM's sequential learning.

* **LSTM Model Summary:**
    ![LSTM Model Summary](path/to/lstm_model_summary.png)
    *Figure 2.3. LSTM Model Summary*

## Dataset
The project utilized a publicly available dataset consisting of dermoscopic images of skin lesions, specifically the "Skin Cancer: Malignant vs. Benign dataset". The dataset comprises 3,300 images, evenly split between benign and malignant categories. Images underwent preprocessing steps including normalization, resizing to 224x224 pixels, and augmentation.

## Experiments & Results

### CNN Model Results
The CNN model demonstrated strong performance and good generalization capabilities.
* **Validation Accuracy:** 85.53%
* **Validation Loss:** 0.4094
* **Test Accuracy:** 82.73%
* **Test Loss:** 0.3881
* **Processing Speed:** 35 ms/step for validation, 68 ms/step for testing

* **CNN Model Training and Validation Accuracy:**
    ![CNN Training and Validation Accuracy](path/to/cnn_accuracy_plot.png)
    *Figure 3.1. CNN Model Training and Validation Accuracy*
    The training accuracy reached approximately 85%, while validation accuracy stabilized around 83%.

* **CNN Model Training and Validation Loss:**
    ![CNN Training and Validation Loss](path/to/cnn_loss_plot.png)
    *Figure 3.2. CNN Model Training and Validation Loss*
    Training loss steadily decreased, but validation loss showed volatility after epoch 15, indicating some overfitting.

* **CNN Model Evaluation Results:**
    ![CNN Evaluation Results](path/to/cnn_evaluation_results.png)
    *Figure 3.3. CNN Model Evaluation Results*

### LSTM Model Results
The LSTM model, particularly with ResNet50 feature extraction, showed competitive performance across evaluation sets.
* **Validation Accuracy:** 81.38% (reported as 82% rounded)
* **Validation Loss:** 0.7669
* **Test Accuracy:** 86.09% (reported as 85% rounded)
* **Test Loss:** 0.5256
* **Processing Speed:** 9 ms/step for validation, 10 ms/step for testing

* **LSTM Model Training and Validation Accuracy:**
    ![LSTM Training and Validation Accuracy](path/to/lstm_accuracy_plot.png)
    *Figure 3.4. LSTM Model Training and Validation Accuracy*
    Training accuracy reached 97%, but validation accuracy fluctuated between 72% and 85%, indicating significant overfitting.

* **LSTM Model Training and Validation Loss:**
    ![LSTM Training and Validation Loss](path/to/lstm_loss_plot.png)
    *Figure 3.5. LSTM Model Training and Validation Loss*
    Training loss steadily decreased, while validation loss showed extreme volatility with dramatic spikes, confirming severe overfitting.

* **LSTM Model Evaluation Results:**
    ![LSTM Evaluation Results](path/to/lstm_evaluation_results.png)
    *Figure 3.6. LSTM Model Evaluation Results*

### RNN Model Results
The Simple RNN model exhibited modest performance due to its simpler architecture and limited ability to capture complex spatial features in images.
* **Validation Accuracy:** Approximately 67.45%
* **Validation Loss:** 0.5994
* **Consistent Validation Accuracy:** ~68.74%

* **RNN Model Evaluation Results:**
    ![RNN Evaluation Results](path/to/rnn_evaluation_results.png)
    *Figure 3.7. RNN Model Evaluation Results*

## Discussion
The comparative analysis revealed that the LSTM model achieved the highest test accuracy (85%) by combining ResNet50's feature extraction with LSTM's sequential processing, though it showed tendencies of overfitting. The CNN model demonstrated consistent performance with an 82.73% test accuracy and stable learning curves, making it a reliable choice for practical applications. The Simple RNN model had the lowest accuracy (67.45%) but offered computational efficiency, which could be beneficial in resource-constrained environments.

## Conclusion
This project successfully implemented and compared CNN, LSTM, and Simple RNN models for skin cancer classification. The ResNet50+LSTM hybrid approach achieved the highest accuracy, showcasing the benefits of transfer learning and sequential processing. The CNN model offered a strong balance of accuracy and computational efficiency, making it suitable for practical applications. While the Simple RNN had lower accuracy, its efficiency could suit limited environments. Overall, the system shows significant promise in aiding early and accurate skin cancer detection, though further clinical validation and enhancements for overfitting and generalization are crucial for real-world reliability.

## References
1.  Skin Cancer Detection Using CNN. Retrieved from [https://medium.com/ai-techsystems/skin-cancer-detection-using-cnn-7ba3ca8d3dc3](https://medium.com/ai-techsystems/skin-cancer-detection-using-cnn-7ba3ca8d3dc3)
2.  Exploring ResNet50: An In-Depth Look at the Model Architecture and Code Implementation. Retrieved from [https://medium.com/@nitishkundu](https://medium.com/@nitishkundu)
3.  Implementation of Long Short-Term Memory (LSTM). Retrieved from [https://medium.com/@pennQuin/implementation-of-long-short-term-memory-lstm-81e35fa5ca54](https://medium.com/@pennQuin/implementation-of-long-short-term-memory-lstm-81e35fa5ca54)
4.  Recurrent Neural Networks (RNN): From Basic to Advanced. Retrieved from [https://medium.com/@sachinsoni600517/recurrent-neural-networks-rnn-from-basic-to-advanced-1da22aafa009](https://medium.com/@sachinsoni600517/recurrent-neural-networks-rnn-from-basic-to-advanced-1da22aafa009)

## Usage
To run this project:
1.  Ensure you have Python and necessary libraries (TensorFlow, Keras, NumPy, OpenCV, etc.) installed.
2.  Download the "Skin Cancer: Malignant vs. Benign" dataset.
3.  Execute the Jupyter notebooks (`Skin_Cancer_Detection_using_CNN.ipynb`, `Skin_Cancer_Detection_using_LSTM.ipynb`, `Skin_Cancer_Detection_using_RNN.ipynb`) to replicate the experiments.
