# ContexGuard Framework

A Modular, Context-Aware Adaptive Security Framework for Dynamic Edge Environments.

## Run Locally

1. Install requirements: pip install -r requirements.txt
2. Run the app: python app/main.py

## Run with Docker

docker-compose up --build

## AI Model Training and Setup



ContexGuard uses a Machine Learning policy plugin driven by a Random Forest classifier. Because the repository does not include the pre-trained binary (`.joblib`) due to size and security best practices, you must compile the dataset and train the model locally before using the `MLPolicy`.



\### Step 1: Create the Training Script

Create a new file named `train\_rf\_model.py` inside the `research/` directory. This script will generate the 100,000 synthetic contextual samples  and train the classifier with the exact hyperparameters used in the framework's evaluation (100 estimators, max depth 10).



```python

\# research/train\_rf\_model.py

import os

import numpy as np

import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model\_selection import train\_test\_split

from sklearn.metrics import classification\_report



def generate\_training\_data(num\_samples=100000):

&nbsp;   print(f"Generating {num\_samples} synthetic context samples...")

&nbsp;   X = \[]

&nbsp;   y = \[]

&nbsp;   

&nbsp;   for \_ in range(num\_samples):

&nbsp;       # 15% probability of generating an anomalous/attack sample

&nbsp;       is\_attack = np.random.rand() < 0.15 

&nbsp;       

&nbsp;       if is\_attack:

&nbsp;           device\_score = np.random.uniform(0.0, 0.4)

&nbsp;           loc\_trust = 0.0  # e.g., remote\_untrusted

&nbsp;           time\_anomaly = np.random.uniform(0.6, 1.0)

&nbsp;           net\_anomaly = np.random.uniform(0.7, 1.0)

&nbsp;           label = 1  # 1 = Deny/Malicious

&nbsp;       else:

&nbsp;           device\_score = np.random.uniform(0.7, 1.0)

&nbsp;           loc\_trust = 1.0  # e.g., office

&nbsp;           time\_anomaly = np.random.uniform(0.0, 0.3)

&nbsp;           net\_anomaly = np.random.uniform(0.0, 0.2)

&nbsp;           label = 0  # 0 = Permit/Legitimate

&nbsp;           

&nbsp;       # Add some random noise to make the model robust

&nbsp;       device\_score = np.clip(device\_score + np.random.normal(0, 0.05), 0, 1)

&nbsp;       net\_anomaly = np.clip(net\_anomaly + np.random.normal(0, 0.05), 0, 1)

&nbsp;       

&nbsp;       # Feature array: \[device\_score, location\_trust, time\_anomaly, network\_anomaly]

&nbsp;       X.append(\[device\_score, loc\_trust, time\_anomaly, net\_anomaly])

&nbsp;       y.append(label)

&nbsp;       

&nbsp;   return np.array(X), np.array(y)



def train\_and\_save\_model():

&nbsp;   # 1. Generate Data

&nbsp;   X, y = generate\_training\_data(100000)

&nbsp;   X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=42)

&nbsp;   

&nbsp;   # 2. Train Model

&nbsp;   print("Training Random Forest Classifier (n\_estimators=100, max\_depth=10)...")

&nbsp;   clf = RandomForestClassifier(n\_estimators=100, max\_depth=10, random\_state=42)

&nbsp;   clf.fit(X\_train, y\_train)

&nbsp;   

&nbsp;   # 3. Evaluate Model

&nbsp;   print("\\nModel Evaluation on Test Set:")

&nbsp;   y\_pred = clf.predict(X\_test)

&nbsp;   print(classification\_report(y\_test, y\_pred, target\_names=\['Permit (0)', 'Deny (1)']))

&nbsp;   

&nbsp;   # 4. Save Model to Plugins Directory

&nbsp;   target\_dir = "../plugins/ai\_models"

&nbsp;   os.makedirs(target\_dir, exist\_ok=True)

&nbsp;   model\_path = os.path.join(target\_dir, "rf\_risk\_model.joblib")

&nbsp;   

&nbsp;   joblib.dump(clf, model\_path)

&nbsp;   print(f"\\nSuccess! Model saved to: {model\_path}")



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   train\_and\_save\_model()

