import eel
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

xgb = XGBClassifier(n_estimators= 500 , max_depth= 3 , learning_rate = 0.1)
xgb = joblib.load("classifer.model")
sc = StandardScaler()
sc = joblib.load("scaler.model")


def get_feature_vector(age, bmi, smoker, drinker, stroke, pHealth, mHealth, walking, sex, race, diabetic, exercise, gHealth, sleep, asthma, kidney, skin):
    arr = []
    arr.append(float(bmi))
    arr.append(int(smoker))
    arr.append(int(drinker))
    arr.append(int(stroke))
    arr.append(int(pHealth))
    arr.append(int(mHealth))
    arr.append(int(walking))
    arr.append(int(sex))
    arr.append(int(age))
    arr.append(int(exercise))
    arr.append(int(gHealth))
    arr.append(int(sleep))
    arr.append(int(asthma))
    arr.append(int(kidney))
    arr.append(int(skin))

    if int(diabetic) == 0:
        arr.append(1)
        arr.append(0)
        arr.append(0)
        arr.append(0)
    elif int(diabetic) == 1:
        arr.append(0)
        arr.append(0)
        arr.append(1)
        arr.append(0)

    if int(race) == 1:
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(1)
    elif int(race) == 2:
        arr.append(0)
        arr.append(0)
        arr.append(1)
        arr.append(0)
        arr.append(0)
        arr.append(0)
    elif int(race) == 3:
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(1)
        arr.append(0)
        arr.append(0)
    elif int(race) == 4:
        arr.append(0)
        arr.append(1)
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(0)
    elif int(race) == 5:
        arr.append(1)
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(0)
    elif int(race) == 6:
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(0)
        arr.append(1)
        arr.append(0)

    x1 = np.array(arr)
    print(x1)
    x1 = sc.transform(x1.reshape(1, -1))
    print(x1)

    return x1



eel.init("web")

@eel.expose
def makePrediction(age, bmi, smoker, drinker, stroke, pHealth, mHealth, walking, sex, race, diabetic, exercise, gHealth, sleep, asthma, kidney, skin):
    features = get_feature_vector(age, bmi, smoker, drinker, stroke, pHealth, mHealth, walking, sex, race, diabetic, exercise, gHealth, sleep, asthma, kidney, skin)
    pred = xgb.predict_proba(features)
    print(pred)
    print("Returning: ", pred[0][0])
    return str(pred[0][0])
@eel.expose
def hello():
    print("Hello World")

eel.start("index.html")