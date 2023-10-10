# ML-Binary-classification-fungi
Binary classification of fungi using KNN, DNN and Decision Tree algorithms

The dataset includes descriptions of hypothetical specimens corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota fungi families extracted from the Audubon Society Field Guide to North American Mushrooms. Each species is identified as definitely edible, definitely poisonous or with unknown and not recommended edibility.
About the file.
The dataset used was downloaded from the KAGGLE website:
https://www.kaggle.com/datasets/uciml/mushroom-classification
<pre>
Information about attributes: (classes: edible=e, poisonous=p)
• head-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
• head-surface: fibrous=f, grooves=g, scaly=y, smooth=s
• head-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
• bruises: bruises=t,no=f
• smell: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
• gill-attachment: attached=a, descending=d, free=f, notched=n
• gill-spacing: close=c,crowded=w,distant=d
• gill-size: broad=b, narrow=n
• gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow =y
• stalk-shape: enlarging=e, tapering=t
• stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
• stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
• stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
• stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
• stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
• veil-type: partial=p, universal=u
• veil-color: brown=n, orange=o, white=w, yellow=y
• ring-number: none=n, one=o, two=t
• ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
•	spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
•	population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
•	habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

  _______________________________________________________________________
In: 
df.head()
Out: 
  class cap-shape cap-surface  ... spore-print-color population habitat
0     p         x           s  ...                 k          s       u
1     e         x           s  ...                 n          n       g
2     e         b           s  ...                 n          n       m
3     p         x           y  ...                 k          s       u
4     e         x           s  ...                 n          a       g
_______________________________________________________________________
In:
df.describe()
Out:
       class cap-shape cap-surface  ... spore-print-color population habitat
count   8124      8124        8124  ...              8124       8124    8124
unique     2         6           4  ...                 9          6       7
top        e         x           y  ...                 w          v       d
freq    4208      3656        3244  ...              2388       4040    3148
___________________________________________________________________
In: 
	df.info()
Out:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8124 entries, 0 to 8123
Data columns (total 23 columns):
 #   Column                    Non-Null Count  Dtype 
---  ------                    --------------  ----- 
 0   class                     8124 non-null   object
 1   cap-shape                 8124 non-null   object
 2   cap-surface               8124 non-null   object
 3   cap-color                 8124 non-null   object
 4   bruises                   8124 non-null   object
 5   odor                      8124 non-null   object
 6   gill-attachment           8124 non-null   object
 7   gill-spacing              8124 non-null   object
 8   gill-size                 8124 non-null   object
 9   gill-color                8124 non-null   object
 10  stalk-shape               8124 non-null   object
 11  stalk-root                8124 non-null   object
 12  stalk-surface-above-ring  8124 non-null   object
 13  stalk-surface-below-ring  8124 non-null   object
 14  stalk-color-above-ring    8124 non-null   object
 15  stalk-color-below-ring    8124 non-null   object
 16  veil-type                 8124 non-null   object
 17  veil-color                8124 non-null   object
 18  ring-number               8124 non-null   object
 19  ring-type                 8124 non-null   object
 20  spore-print-color         8124 non-null   object
 21  population                8124 non-null   object
 22  habitat                   8124 non-null   object
dtypes: object(23)
memory usage: 1.4+ MB
_______________________________________________________________________
In: 
for col in df.columns:
    		print(col, " : ", df[col].unique())
Out: 
class  :  ['p' 'e']
cap-shape  :  ['x' 'b' 's' 'f' 'k' 'c']
cap-surface  :  ['s' 'y' 'f' 'g']
cap-color  :  ['n' 'y' 'w' 'g' 'e' 'p' 'b' 'u' 'c' 'r']
bruises  :  ['t' 'f']
odor  :  ['p' 'a' 'l' 'n' 'f' 'c' 'y' 's' 'm']
gill-attachment  :  ['f' 'a']
gill-spacing  :  ['c' 'w']
gill-size  :  ['n' 'b']
gill-color  :  ['k' 'n' 'g' 'p' 'w' 'h' 'u' 'e' 'b' 'r' 'y' 'o']
stalk-shape  :  ['e' 't']
stalk-root  :  ['e' 'c' 'b' 'r' '?']
stalk-surface-above-ring  :  ['s' 'f' 'k' 'y']
stalk-surface-below-ring  :  ['s' 'f' 'y' 'k']
stalk-color-above-ring  :  ['w' 'g' 'p' 'n' 'b' 'e' 'o' 'c' 'y']
stalk-color-below-ring  :  ['w' 'p' 'g' 'b' 'n' 'e' 'y' 'o' 'c']
veil-type  :  ['p']
veil-color  :  ['w' 'n' 'o' 'y']
ring-number  :  ['o' 't' 'n']
ring-type  :  ['p' 'e' 'l' 'f' 'n']
spore-print-color  :  ['k' 'n' 'u' 'h' 'w' 'r' 'o' 'y' 'b']
population  :  ['s' 'n' 'a' 'v' 'y' 'c']
habitat  :  ['u' 'g' 'm' 'd' 'p' 'w' 'l']
</pre>

1. <b>Implementation with KNN algorithm</b> KNN is a simple, supervised machine learning algorithm that can be used for classification or regression tasks. It is based on the idea that the observations closest to a given data point are the most "similar" observations in a data set, and therefore we can classify the unexpected points based on the values of the closest existing points. By choosing K, the user can select the number of observations to use.

Description:
- We read a CSV file in DataFrame
- We load the attributes from the DataFrame into a list
- We split the DataFrame into two separate DataFrames, representing the input and the output
- We encode the labels with LabelEncoder
- We transform the DataFrame into an array with only values without the axis labels
- We define a transformation function (depending on the Encoder used in training) of the new data that will be used for predictions.
- We split the dataset into training and testing subsets
- We calculate the confusion matrix to evaluate the accuracy


<pre>    
Confusion Matrix:
[[1615    7]
[   4 1624]]
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      1622
           1       1.00      1.00      1.00      1628
    accuracy                           1.00      3250
   macro avg       1.00      1.00      1.00      3250
weighted avg       1.00      1.00      1.00      3250
Accuracy: 0.9966153846153846

We introduce new data:
In: 
a = transform_new_data(
['x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u'])
Out: 
	[5, 2, 4, 1, 6, 1, 0, 1, 4, 0, 3, 2, 2, 7, 7, 0, 2, 1, 4, 2, 3, 5]
In:
knn.predict([a])
Out:	
 array([1])
In:
	le_y.inverse_transform([1])
Out: 
	array(['p'], dtype=object)
  
</pre>

2. <b>Implementation with DNN algorithm</b>
- neural network using the TensorFlow platform/framework and the default Keras library
- Three dense layers were used and for the output the single-knot technique is used.
- The first layer has 22 nodes, the second has 11 nodes and the last layer has 1 node with sigmoid activation function to handle the binary classification problem.
- The result of the sigmoid activation function will have a value between 0 and 1 and the output will be interpreted as the probability of the class that is coded with 1 as follows:
- Output>0.5 is considered from the p=poisonous class (coded as 1)
- Output<0.5 is considered from the e=edible class (coded as 0)

<pre>
Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 22)                506       
                                                                 
 dense_1 (Dense)             (None, 11)                253       
                                                                 
 dense_2 (Dense)             (None, 1)                 12        
                                                                 
=================================================================
Total params: 771
Trainable params: 771
Non-trainable params: 0
_________________________________________________________________
None
102/102 - 0s - loss: 0.0159 - accuracy: 0.9969 - 334ms/epoch - 3ms/step
0.9969230890274048

IN: from tensorflow.keras.utils import plot_model
      plot_model(model, to_file='model1.png')
Out[3]: 
<IPython.core.display.Image object>)

In:
model.predict(tf.expand_dims(tf.convert_to_tensor(X_test[3]),0))
Out:
1/1 [==============================] - 0s 27ms/step
array([[0.971193]], dtype=float32)
  
In:
model.predict(tf.expand_dims(tf.convert_to_tensor(X_test[5]),0))
Out:
1/1 [==============================] - 0s 45ms/step
array([[0.9997364]], dtype=float32)
  
In:
model.predict(tf.expand_dims(tf.convert_to_tensor(X_test[10]),0))
Out:
1/1 [==============================] - 0s 30ms/step
array([[0.00472749]], dtype=float32)

</pre>

3. <b>DecisionTree algorithm</b>
Decision trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules derived from data characteristics. A tree can be seen as a constant piecewise approximation.
Viewing the decision tree:
<pre>
  In:
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Mushrooms_Tree.png')
Image(graph.create_png())
Out:
</pre>
![image](https://github.com/TrifanLucian/ML-Binary-classification-fungi/assets/111199896/bc26b29e-ee43-47a3-b0ba-2a40d0a37f25)
Results:
<pre>
  Confusion Matrix:
[[1236    0]
 [   0 1202]]
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      1236
           1       1.00      1.00      1.00      1202
    accuracy                           1.00      2438
   macro avg       1.00      1.00      1.00      2438
weighted avg       1.00      1.00      1.00      2438
Accuracy: 1.0

Tests:
In:
X_test.iloc[1].values.tolist()
Out:
[5, 0, 5, 0, 1, 1, 1, 1, 9, 0, 1, 2, 2, 7, 7, 0, 2, 1, 4, 3, 4, 0]

In:
y_test.iloc[1]
Out:
1
  
In:
clf.predict([[5, 0, 5, 0, 1, 1, 1, 1, 9, 0, 1, 2, 2, 7, 7, 0, 2, 1, 4, 3, 4, 0]])
Out:
array([1])
</pre>
