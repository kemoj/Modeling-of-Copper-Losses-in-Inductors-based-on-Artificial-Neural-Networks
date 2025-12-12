**Modeling-of-Copper-Losses-in-Inductors-based-on-Artificial-Intelligence-Techniques**

<img width="465" height="252" alt="image" src="https://github.com/user-attachments/assets/8269b051-9571-4564-aba8-70b17d82aad8" />


This project builds and compares multiple machine-learning regression models to predict solid copper losses (SolidLoss [W]) based on parametric sweeps of frequency, gap size, and winding offset.
The workflow includes data loading, preprocessing, model training, hyperparameter tuning, evaluation, and visualization of results.

 **#Dataset**

The dataset used is Parametric_Sweep_Loss_Table.csv, containing:
Column	Description
freq_khz [kHz]	Input excitation frequency in kHz
gap_size [mm]	Air gap size between magnetic cores
winding_x_offset [m]	Horizontal displacement of the winding
Freq [kHz]	Secondary frequency feature
SolidLoss [W]	Target variable — predicted copper solid losses


**Models Implemented**

The following regression models were trained:

Linear Regression
Polynomial Regression (2nd order)
Support Vector Regression (RBF kernel)
K-Nearest Neighbors Regressor
Random Forest Regressor
Gradient Boosting Regressor (GridSearch-tuned)

*Gradient Boosting was optimized using GridSearchCV over learning rate, depth, estimators, and subsampling*

**Results Summary**

After training and evaluation on the test set, model performance (ranked by R² score) is:
Model	RMSE	R²
Gradient Boosting (tuned)	0.0681	0.99991
Random Forest	0.2525	0.99875
KNN Regressor	3.8423	0.71008
SVR (RBF)	4.2852	0.63940
Polynomial Regression	4.5898	0.58630
Linear Regression	5.9460	0.30572

*Gradient Boosting achieves near-perfect accuracy, capturing the highly nonlinear relationship between the geometric/electromagnetic parameters and copper losses*

**Visualizations Generated**
1. Model Comparison Plots
R² bar chart → model_comparison_R2.pdf
RMSE bar chart → model_comparison_RMSE.pdf

2. True vs Predicted Plots
For SVR model
For the top 3 performing models (Gb, RF, KNN)
Each saved as separate PDFs.

3. Residual Distribution Plot
Residual histogram and KDE for the best model
Saved as: residuals_gradient_boosting_tuned.pdf


**Key Insights**
Copper loss behavior is strongly nonlinear, making tree-based ensemble models far superior to linear or kernel methods.
Gradient Boosting, when tuned, captures subtle geometric-electromagnetic interactions with extremely high fidelity.
Traditional analytical loss equations may miss these fine-scale nonlinearities, making ML a powerful modeling tool.
