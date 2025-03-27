# Titanic-Survival-Prediction
he Titanic Survival Prediction project is a machine learning-based analysis that predicts whether a passenger survived the Titanic disaster based on various features such as age, gender, class, fare, and more. 
## Repository Structure
```
Titanic-Survival-Prediction/
│-- data/
│   │-- train.csv
│   │-- test.csv
│-- notebooks/
│   │-- data_exploration.ipynb
│   │-- model_training.ipynb
│-- src/
│   │-- data_preprocessing.py
│   │-- model.py
│-- results/
│   │-- predictions.csv
## Dataset
The dataset used in this project is the Titanic dataset from Kaggle:https://www.kaggle.com/datasets/yasserh/titanic-dataset
## Project Workflow
1. **Data Exploration**: Analyze and visualize the dataset.
2. **Data Preprocessing**: Handle missing values, feature engineering, and encoding categorical variables.
3. **Model Training**: Train various machine learning models like Logistic Regression, Random Forest, and XGBoost.
4. **Evaluation & Prediction**: Evaluate model performance and generate predictions on the test dataset.

## Installation
To run this project locally, follow these steps:

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run Jupyter Notebook to explore and train models:
```bash
jupyter notebook
```
## Dependencies
The following libraries are required to run the project:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

## Usage
- Run `notebooks/data_exploration.ipynb` to analyze the dataset.
- Run `notebooks/model_training.ipynb` to preprocess data and train models.
- The final predictions are stored in `results/predictions.csv`.

## Results
- The model achieves an accuracy of approximately **XX%** on the validation dataset.
- Feature importance analysis suggests that **Pclass, Sex, and Age** are key predictors of survival.

## Contributing
Feel free to fork this repository and improve the model. Contributions are welcome!

## License
This project is open-source and available under the [MIT License](LICENSE).
