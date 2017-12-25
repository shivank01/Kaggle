import pandas as pd
import numpy as np 
from sklearn import preprocessing,svm,model_selection
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('train.csv')
df1=pd.read_csv('test.csv')
#print(df.head())
#print(df.describe())
df.drop('PassengerId',1,inplace=True)
df.drop('SibSp',1,inplace=True)
df.drop('Parch',1,inplace=True)
df.drop('Embarked',1,inplace=True)

df.fillna(value=-99999, inplace=True)

df1.drop('SibSp',1,inplace=True)
df1.drop('Parch',1,inplace=True)
df1.drop('Embarked',1,inplace=True)

df1.fillna(value=-99999, inplace=True)



def handle_non_numeric_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_values={}
        def convert_to_int(val):
            return text_digit_values[val]


        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            columns_contents=df[column].values.tolist()
            unique_elements=set(columns_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique]=x
                    x+=1

            df[column]=list(map(convert_to_int,df[column]))

    return df


df=handle_non_numeric_data(df)
df1=handle_non_numeric_data(df1)
#print(df['Survived'])

X=np.array(df.drop('Survived',1)).astype(float)
df=df['Survived']
y=np.array(df).astype(int)
#print(y.shape)
#X=handle_non_numeric_data(X)

X_predict=np.array(df1).astype(float)

#X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.1,random_state=0)

clf=svm.SVC(kernel="linear",C=10)
clf.fit(X,y)

# accuracy=clf.score(X_test,y_test)
# print("accuracy:",accuracy)

predict=np.array(df1.drop('PassengerId',1))
predict=preprocessing.scale(predict)
predict=clf.predict(predict)

predict=pd.DataFrame(predict,index=df1['PassengerId'],columns=['Survived'])
print(predict)
predict.to_csv('submit.csv')
