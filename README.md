# machine-learning-challenge

## Challenge Details

This challenge was to create and compare different machine learning models that are capable of classifying different types of exoplanents (whether they are CANDIDATES, FALSE POSITIVES, or CONFIRMED).

Details about the data can be found here: [Kepler Exoplanent Information](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

### Libraries used

[sklearn](https://scikit-learn.org/stable/)
[TensorFlow](https://www.tensorflow.org/guide/keras/sequential_model)

### Cleaning and Preprocessing the Data

To prepare the data to be used in the models, multiple steps were taken. First, null columns and rows were dropped. Next, each column was analyzed to determine if it was needed to make an accurate model. Column koi_teq was dropped, as it was redunant with koi_insol - both these columns are ways to describe the equilibrium temperature. Column koi_insol was kept, as the dataset also included the error associated with this reading and seemed to be a more accurate representation of this parameter. Column koi_disposition was used for the y values, or target, as this is the classification type the model is intended to predict. Finally, the data was split into training data and testing data.

Next, the data was preprocessed in the following ways: First, the X data was scaled using sklearn's MinMaxScaler, which scales the numerical data to a specific range. This is done on both the X_train and X_test data, which is the numerical data in this challenge.

For the neural network and deep learning model, one additional step was taken: the y data was transformed into a binary type of output. In this case, \[1 0 0] would represent CANDIDATE, \[0 1 0] would represent CONFIRMED, and \[0 0 1] would represent FALSE POSITIVE.

### About the Models

Once the data cleaning and preprocessing was completed, different models were created and trained with the x and y training data. Then hyperparameter tuning was employed using GridSearchCV, to see if the models could be improved. For the GridSearchCV tuning, parameters C and gamma were varied to find the best combination that could be used to get the best scored on the test data.

* Three types of SVC models were attempted and their accuracies were compared:
    * Linear SVC model - this model started with a score on test data of ~84% after being trained. Using GridSearchCV, the model's score on test data increased to approximately 88%. In general, score results got better as C increased and gamma decreased. However, computation time also increased as these changes were made (depending on the C and gamma values, the model took up to ~10min to run only 45 fits). 
    * RBF SVC model - this model started with a score on test data of ~82% after being trained. Using GridSearchCV, the model's score on test data increased to approximately 90%. Even using multiple C and gamma parameters in the tuning process did not drastically increase the computation time (the model was able to complete the analysis of 80 fits in less than 1 min).
    * Poly SVC model - this model started with a score on test data of ~84% after being trained. Attempting the GridSearchCV tuning proved to be difficult, as this model ran significantly slower than the previous 2. The number of C and gamma values used was reduced from 3 or 4 values each to just 2 values (20 fits). Depending on the gamma used, the tuning runtime increased dramatically, taking up to 10min per fit. As gamma values decreased, the runtime decreased. Even with these longer runtimes, the overall scores did not increase significantly. Overall this model proved to fall in line with the Linear and RBF models, ending up with a testing score of approximately 89%.

The other models that were tested included a neural network model and deep learning model.
    * Neural Network model - this model was a Sequential model using dense layers. The number of input parameters was 39 (the X data contains 39 columns of numerical information). The number of units was varied to see how the model accuracy would change. Changing from 75 to 100 to 200 did not have any significant impact on the accuracy or loss of this model. The accuracy for the training and test data was approximately 89% and the loss remained at about 25%. Increasing the epochs from 100 to 150 or 200 did increase the training and testing accuracy slightly, but did not have much of an impact on the test loss, and therefore may not be very beneficial.
    * Deep learning model - the only difference between the neural network and this model is that an additional hidden layer was added. In this instance, if the number of epochs was increased (for example for 100 to 150), the training accuracy and loss improved, but the testing loss increased noticeably, indicating that the model was overfit and would not be efficient when using new data.
    
   