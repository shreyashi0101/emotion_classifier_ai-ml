Description
This project builds a complete machine learning pipeline that classifies emotional states from audio clips using speech and song recordings. Audio features like MFCCs, Chroma, and Spectrograms were extracted and used to train a classifier. The final model—a Random Forest Classifier—was tuned to achieve high accuracy and deployed as a Streamlit web app.

Dataset
We used two datasets:
Audio_Speech_Actors_01-24: Contains speech audio files.
Audio_Song_Actors_01-24: Contains sung audio .
Each audio filename encodes metadata, including emotion type, which was mapped as:
emotion_map = {
'01': 'neutral',
'02': 'calm',
'03': 'happy',
'04': 'sad'
'05': 'angry',
'06': 'fearful',
'07': 'disgust',
'08': 'surprised'
}


Preprocessing & Feature Extraction
Feature Extraction:
We extracted the following features using librosa:
We extracted 40 Mel Frequency Cepstral Coefficients (MFCCs) per audio file.
For each MFCC, we computed:
Mean over time
Standard Deviation over time
These were then concatenated to form a single 80-dimensional feature vector:
40 MFCC means + 40 MFCC standard deviations = 80 features total

Data Cleaning
After extracting MFCC audio features from each .wav file, the resulting dataset was cleaned to ensure it was ready for model training:

Type Conversion: All feature columns (except the label column) were converted to numeric using pandas.to_numeric(), forcing non-numeric entries to NaN:
Missing Value Handling: Any rows containing NaN values were dropped:
Index Reset: After dropping rows, the DataFrame index was reset

 Data Exploration & Balancing
To better understand the distribution and structure of our audio feature dataset, the following methods were applied:

Class Distribution Visualization: A count plot was generated to visualize how emotions were distributed across the dataset. This revealed class imbalances, with certain emotions being overrepresented while others were underrepresented.

Feature Distributions (MFCCs): Histograms of MFCC features were plotted in batches to analyze their individual distributions. This helped identify patterns, scale ranges, and possible outliers in the audio-derived features.

Class Balancing: To ensure fair and unbiased model training, the dataset was balanced using downsampling:

The minimum number of samples among all emotion classes was identified.

Each class was randomly downsampled to this minimum count.

The final balanced dataset was then shuffled to ensure random ordering.

These preprocessing steps ensured that the model would not be biased toward any specific emotion class and that the feature space was well understood before training.



Label Encoding:

Used LabelEncoder to convert string labels (e.g., "happy") to integers.
Saved using joblib for reuse in prediction.

Train-Test Split:
Combined both speech and song data after feature extraction.
Final dataset shape: (2452, 41) → 40 features + 1 label.
Used an 80/20 split for training and testing.
Stratified sampling was applied to preserve emotion distribution.

Baseline Model Performance (Random Forest)
Using 5-fold cross-validation, the baseline Random Forest classifier achieved:
Average CV Accuracy: 70.88%

On the test set, it reached an overall accuracy of 74.75%.
The model showed strong precision and recall across multiple emotion classes, especially for emotionally distinct labels.

 Alternative Model (Support Vector Machine)
The SVM model underperformed with a test set accuracy of only 50.91%.
While recall for a few classes was high (e.g., label 1), it struggled significantly with others, leading to poor macro-average F1-score.

Optimized Random Forest (Grid Search)
After hyperparameter tuning via GridSearchCV, the best Random Forest model achieved:
Best Cross-Validation Accuracy: 75.63%
Test Set Accuracy: 76.58%
This improvement validates the importance of model tuning, especially in multi-class classification problems.


After evaluating multiple models and tuning hyperparameters, the final Random Forest classifier was trained using the best-performing parameters identified through GridSearchCV. These parameters were:

n_estimators: 200
max_depth: 30
min_samples_leaf: 1
min_samples_split: 2
bootstrap: False
random_state: 42
The model was trained on the full training dataset, and the final training phase completed successfully.

 Model Evaluation
After training the final model, it was evaluated using the test dataset. The evaluation was performed using multiple metrics:
Accuracy: 76.58% on the unseen test data.
Precision, Recall, F1-Score: Computed for each emotion class to assess classification quality.
Confusion Matrix: Visualized to understand misclassification patterns between emotions.
![image](https://github.com/user-attachments/assets/da41734b-d648-49bf-9603-92c6f741ce10)


 Classification Highlights:
Highest Accuracy was observed in calm, angry, and neutral classes.
Challenging Classes included surprised and disgust, likely due to data overlap or fewer training samples.

 Feature Importance Analysis
To understand which features contributed most to emotion classification, a feature importance analysis was performed using the Random Forest model.
MFCC (Mel-Frequency Cepstral Coefficients) were used as features.
The model computed an importance score for each MFCC feature.
A bar plot was generated to visualize these scores in descending order of importance.
Insights:
mfcc_0, mfcc_40, and mfcc_72 were among the most significant features.
Helps in understanding which spectral features are most influential for emotion detection.
![image](https://github.com/user-attachments/assets/40cab0d0-437f-4dd4-93f3-7ead3f524528)



Model Testing with Streamlit
A lightweight Streamlit script (test.py) was created to test the trained model interactively using .wav files.
Features:
Upload a .wav file to test the model's prediction.
Extracts 80 MFCC features using librosa.
Loads the saved RandomForestClassifier model and LabelEncoder.
Predicts and displays the emotion from speech.
This helped verify model performance on unseen audio clips outside the training pipeline.
Note: This is a testing script, not the final deployed app interface.

 Streamlit Web App Interface
The project includes a user-friendly Streamlit web application (app.py) that allows users to upload .wav audio files and get real-time emotion predictions.
Features:
File Upload: Users can upload .wav files via a simple drag-and-drop interface.
Audio Playback: Uploaded audio can be played directly in the browser.
Feature Extraction: 80 MFCC features are extracted using librosa for accurate emotion detection.
Prediction: The trained RandomForestClassifier model and associated LabelEncoder are loaded to make predictions.
Output: The predicted emotion is displayed with a success message in a clean interface.
Files Used:
emotion_classifier_model.pkl: Pre-trained machine learning model.
label_encoder.pkl: Label encoder to map predicted classes to emotion names.
This web app serves as the final front-end interface for the speech emotion classification system.
![image](https://github.com/user-attachments/assets/291ee13c-b449-4819-a9d8-4f652b6116cc)
app url:https://emotionclassifier-wmnrtfb9i2ugs4fnpsrxrq.streamlit.app/ (this is the website url)
