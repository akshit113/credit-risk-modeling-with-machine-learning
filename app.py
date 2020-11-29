import pickle

import uvicorn
from fastapi import FastAPI
from pandas import DataFrame, concat

from credit import Credit

app = FastAPI()
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message: hello stranger!'}


"""
0.766126609	45	2	0.802982129	9120	13	0	6	0	2
{
  "revolving_utilization": 0.76,
  "age": 45,
  "n_30_59_days_past_due": 2,
  "debt_ratio": 0.802982129,
  "monthly_income": 9120,
  "n_open_credit_lines": 13,
  "n_90_days_late": 0,
  "n_real_estate_loans": 6,
  "n_60_89_past_due": 0,
  "n_dependents": 2
}


"""


def get_input(data):
    revolving_utilization = data['revolving_utilization']
    age = data['age']
    n_30_59_days_past_due = data['n_30_59_days_past_due']
    debt_ratio = data['debt_ratio']
    monthly_income = data['monthly_income']
    n_open_credit_lines = data['n_open_credit_lines']
    n_90_days_late = data['n_90_days_late']
    n_real_estate_loans = data['n_real_estate_loans']
    n_60_89_past_due = data['n_60_89_past_due']
    n_dependents = data['n_dependents']
    ls = [[revolving_utilization, age, n_30_59_days_past_due, debt_ratio, monthly_income, n_open_credit_lines,
           n_90_days_late, n_real_estate_loans, n_60_89_past_due, n_dependents]]
    test_df = DataFrame(ls, columns=['RevolvingUtilizationOfUnsecuredLines', 'age',
                                     'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                                     'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                                     'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                                     'NumberOfDependents'])
    return test_df


@app.post('/predict')
def predict_delinquency(data: Credit):
    data = data.dict()
    test_df = get_input(data)
    from model import import_data, normalize_columns, one_hot_encode, clean_data
    df = import_data(fname='datasets/cs-training.csv', train=False)
    df = clean_data(df)
    test_df = clean_data(test_df)
    A = set(test_df.columns)
    B = set(df.columns)
    print(A - B)
    print(B - A)
    fdf = concat([test_df, df])
    normalized = normalize_columns(fdf, colnames=['age', 'MonthlyIncome', 'NumberOfDependents'])
    ohe = one_hot_encode(normalized, test_df, colnames=['ages'])
    inp = ohe.iloc[0:1, :]
    print(type(inp))
    print(inp.shape)

    # pass list of list
    pred = classifier.predict(inp)
    print(str(pred))
    return str(pred)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    print('done')
    pass
