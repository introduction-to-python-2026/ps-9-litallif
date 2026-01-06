# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd
df = pd.read_csv("parkinsons.csv")
y = df["status"]
X = df[["MDVP:Jitter(%)", "HNR"]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,  
    y,           
    test_size=0.2, 
    random_state=42
)
from sklearn.svm import SVC
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
import joblib
import yaml

config_data = {
    "selected_features": ["MDVP:Jitter(%)", "HNR"],  
    "path": "parkinson_model.joblib"               
}

with open("config.yaml", "w") as file:
    yaml.dump(config_data, file)
joblib.dump(model, 'my_model.joblib')

