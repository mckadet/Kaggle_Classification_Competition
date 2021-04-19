# Kaggle_Classification_Competition

## Purpose of the project
The goal of this project was fairly simple: create a model that will successfully predict which category a monster falls into based on their characteristics. In other words, the goal was to create a classification model using regression techniques for a dataset of 900 ghosts, goblins, and ghouls.

## File Descriptions
Classification_RF_Script.R - File containing the code used for the project
Class_Submit_rf1.csv - Submission for Random ForestModel
nn_submission.csv - Submission for xgbTree model for comparison
xgb_submission.csv - Submission for the ensemble model

## Methods for Data Cleaning
For this particular project, the data had already been prepped and cleaned. The data contains 900 observations with several explanatory variables including the color of the monster, their hair length, whether or not they have a soul, and several others. While no feature engineering was done for the random forest and boosting models, feature engineering of sorts was used for the ensemble model which will be explained in greater detail later on.

## Methods for Modeling/Prediction
For the first model, I used a random forest classification regression analysis with 500 trees and repeated cross validation. The model performs fairly well with mostly default tuning parameters and included main effects for all explanatory variables as well as one interaction term for soul and hair length, since an initial exploration of the data revelated a potential relationship between these two.

The second model was similar but performed much worse. This model was an extreme gradient boosting tree regession analysis also with repeated cross validation and the same sets of explanatory variables that were used previously.

For the final model, feature engineering was used to create an ensemble model for taking the best results of 6 different types of modeling techniques including gradient boosting, random forest, svm, knn, neive bayes, neural networks, and logistic regression. The probabilities for classification for each of these models were combined to create a set of explanatory variables containing the likelihood of placement in each of the three levels of the response variable. In this way, this model used the probabilities obtained from each of the 6 prior models for classification. Suprisingly, this model still did not outperform the random forest model that was used first.
