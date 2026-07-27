import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("cardekho_data.csv")
with pd.option_context('display.max_rows',10,'display.max_columns',None):
     print(df.head)
print(df.duplicated().sum())
print(df.isnull().sum())
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.nunique())
print(df["Car_Name"].unique().tolist())
print(df.dtypes)


cate_col = []
for col in df.columns:
    if df[col].dtypes == "object":
        cate_col.append(col)
print(cate_col)

num_col = []
for col in df.columns:
      if col not in cate_col+["Selling_Price"]:
         num_col.append(col)
         
print(num_col)
          
X = df.drop(["Selling_Price"] , axis=1 )       
y = df["Selling_Price"] 

X_train,X_test,y_train,y_test = train_test_split(X, y ,random_state=42, test_size=0.1)
          
preprocessor =  ColumnTransformer(
    transformers= [
        ("cat",(OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)),cate_col)
    ],remainder="passthrough"  
)

model = Pipeline(steps=[
             ("preprocess", preprocessor),
             ("model", GradientBoostingRegressor(  
                     n_estimators=220,
                     max_depth=6,
                     learning_rate=0.06,
                     min_samples_leaf=4,
                     random_state=42,))
    
])

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
# print("SCORE",model.score(y_test,y_pred))
joblib.dump(model, "car_price_pipeline.pkl")