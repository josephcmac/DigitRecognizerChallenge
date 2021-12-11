from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StructField, StructType
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow import convert_to_tensor
from tensorflow import reshape
from numpy import argmax

spark = SparkSession.builder.getOrCreate()
df = spark.read.option("header", True).option("inferSchema", True).csv('train.csv')
L = array([list(train.asDict().values()) for train in df.collect()])
del(df)
x_train = L[:,1:785]
y_train = L[:,0]
del(L)
x_train = [x/255. for x in x_train]
x_train = convert_to_tensor(x_train, dtype="float32")
x_train = reshape(x_train, (x_train.shape[0], 28, 28, 1))
df = spark.read.option("header", True).option("inferSchema", True).csv('test.csv')
x_pred = array([list(train.asDict().values()) for train in df.collect()])
del(df)
x_pred = [x/255. for x in x_pred]
x_pred = convert_to_tensor(x_pred, dtype="float32")
x_pred = reshape(x_pred, (x_pred.shape[0], 28, 28, 1))

for d in range(10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_trainD = [ to_categorical( 1 if y == d else 0 , 2) for y in y_train ]
    y_trainD = convert_to_tensor(y_trainD, dtype="float32")
    model.fit(x_train, y_trainD, batch_size=10, epochs=20, shuffle=True)
    del(y_trainD)
    y_pred = model.predict(x_pred, batch_size=10)
    y_pred = [ y_pred[j][1] for j in range( len(y_pred) ) ]
    y_pred = [(j+1, float(y_pred[j])) for j in range(len(y_pred))]
    output_df = spark.createDataFrame(spark.sparkContext.parallelize(y_pred),  StructType([
            StructField('ImageId', IntegerType(), True),
            StructField('Probability', FloatType(), True)
        ]))
    output_df.toPandas().to_csv('component'+str(d)+'.csv', index=False)

L = []
for d in range(10):
    df = spark.read.option("inferScheme", True).option("header",True).csv('component'+str(d)+'.csv')
    L.append([row.asDict()['Probability'] for row in df.collect()] ) 
del(df)
S = []
for j in range(len(L[0])):
    prob = [ float(L[d][j]) for d in range(10) ]

    S.append( (int(j+1), int(argmax(prob)) ) )
output_df = spark.createDataFrame( spark.sparkContext.parallelize(S), StructType([
    StructField("ImageId", IntegerType(), True),
    StructField("Label", IntegerType(), True)
]))
output_df.toPandas().to_csv('submission.csv', index = False)
