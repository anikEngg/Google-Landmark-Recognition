#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os


# In[2]:


pip install keras.utils 


# In[4]:


##from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


from keras.applications.vgg16 import VGG16


# In[6]:


from keras.applications.vgg19 import VGG19


# In[7]:


from tensorflow.keras.utils import to_categorical


# In[8]:


# Constants
img_rows = 224
img_cols = 224
input_shape = (img_rows,img_cols,3)
epochs = 10
batch_size = 64


# In[9]:


# Get ResNet-50 Model
def getResNet50Model(lastFourTrainable=False):
    resnet_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=True)
    # Make all layers non-trainable
    for layer in resnet_model.layers[:]:
        layer.trainable = False
    # Add fully connected layer which have 1024 neuron to ResNet-50 model
    output = resnet_model.get_layer('avg_pool').output
    output = Flatten(name='new_flatten')(output)
    output = Dense(units=1024, activation='relu', name='new_fc')(output)
    predictions = Dense(units=50, activation='softmax')(output)
    resnet_model = Model(resnet_model.input, predictions)
    # Make last 4 layers trainable if lastFourTrainable == True
    if lastFourTrainable == True:
        resnet_model.get_layer('conv5_block3_2_bn').trainable = True
        resnet_model.get_layer('conv5_block3_3_conv').trainable = True
        resnet_model.get_layer('conv5_block3_3_bn').trainable = True
        resnet_model.get_layer('new_fc').trainable = True
    # Compile ResNet-50 model
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    resnet_model.summary()
    return resnet_model


# In[11]:


traindf = pd.read_csv(r"D:\ML project\LandMark\train.csv")
traindf.head()


# In[12]:


traindf.shape


# In[13]:


landmark_unique = traindf['landmark_id'].unique()
len(landmark_unique)


# In[14]:


landmark_unique[0:50]


# In[15]:


image_ids = []
labels = []
temp_labels = []
i=0
for id_ in landmark_unique[0:50]:
    for iid in traindf['id'][traindf['landmark_id'] == id_]:
        image_ids.append(iid)
        labels.append(id_)
        temp_labels.append(i)
    i = i+1
len(image_ids)


# In[31]:


pip install kaggle


# In[26]:


import pandas as pd


# Import Kaggle API and set the competition name
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Set competition name
competition_name = 'Google Landmark Recognition 2021'

# List files in the competition
files = api.competition_list_files(competition_name)

# Get the file name you want to import
file_name = 'train.csv'  # Adjust this according to your requirements

# Import the file into a pandas DataFrame without downloading it locally
data = api.competition_download_file(Google Landmark Recognition 2021, kaggle, path='"D:\ML project\LandMark\kaggle.json"')

# Read the CSV file into a DataFrame
df = pd.read_csv(r"D:\ML project\LandMark\train.csv")


# In[29]:


mainpath = "kaggle competitions download -c landmark-recognition-2021"
image_path = []
images_pixels = []

for i in range(0,len(image_ids)):
    first_dir = os.path.join(mainpath,image_ids[i][0])
    second_dir = os.path.join(first_dir,image_ids[i][1])
    third_dir = os.path.join(second_dir,image_ids[i][2])
    finalpath = os.path.join(third_dir,image_ids[i]+'.jpg')
    
    img_pix = cv2.imread(finalpath,1)
    images_pixels.append(cv2.resize(img_pix, (224,224)))
    
    image_path.append(finalpath)


# In[32]:


files.upload()


# In[ ]:


print(temp_labels)


# In[ ]:


import random


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(16, 16)

next_pix_ = image_path

for i, img_path in enumerate(next_pix_[0:16]):
    
    sp = plt.subplot(5, 4, i + 1)
    sp.axis('Off')

    img = cv2.imread(img_path)
    plt.imshow(img)

plt.show()


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(16, 16)

next_pix = image_path
random.shuffle(next_pix)

for i, img_path in enumerate(next_pix[0:12]):
    
    sp = plt.subplot(4, 4, i + 1)
    sp.axis('Off')

    img = cv2.imread(img_path)
    plt.imshow(img)

plt.show()


# In[ ]:


shuf = list(zip(images_pixels,temp_labels))
random.shuffle(shuf)

train_data, labels_data = zip(*shuf)
print('Images: ', len(train_data))
print('Image labels: ', len(labels_data))


# In[ ]:


print(labels_data)


# In[ ]:


train_data = np.array(train_data) #/ 255


# In[ ]:


a = np.array(labels_data)
a_pd = pd.get_dummies(a).astype('float32').values 


# In[ ]:


print(labels_data[0:6])


# In[ ]:


print(a_pd)


# In[ ]:


labels_data=a_pd


# In[ ]:


print(labels_data[0:5])


# In[ ]:


labels_data.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_data, labels_data, test_size = 0.3, random_state=101)

print("X train data : ", len(X_train))
print("X label data : ", len(X_test))
print("Y test data : ", len(Y_train))
print("Y label data : ", len(Y_test))


# In[ ]:


num_sample=X_train.shape[0]
#20% of the enteries has to be in validation set
validation_freq=int(num_sample*0.2)
#Generating random sample of indices equal to validation_freq
validationlist = random.sample(range(0, num_sample), validation_freq)
traininglist=list(set(range(0,num_sample))-set(validationlist))
print("No interesection between validationlist and traininglist:",set(traininglist).intersection(validationlist))


# In[ ]:


train_x_final=[]
train_y_final=[]
for val in traininglist:
    train_x_final.append(X_train[val])
    train_y_final.append(Y_train[val])
train_x_final=np.array(train_x_final)
train_y_final=np.array(train_y_final)
print("Training data shape",train_x_final.shape)
print("Training label shape",train_y_final.shape)


# In[ ]:


validation_x_final=[]
validation_y_final=[]
for val in validationlist:
    validation_x_final.append(X_train[val])
    validation_y_final.append(Y_train[val])
validation_x_final=np.array(validation_x_final)
validation_y_final=np.array(validation_y_final)
print("Validation data shape",validation_x_final.shape)
print("Validation label shape",validation_y_final.shape)


# In[ ]:


# plot the training results
def plot_hist(history,title):
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(str(title)+' accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(str(title)+' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[ ]:


# Get VGG-16 Model
def getVGG16Model(lastFourTrainable=False):
    vgg_model = VGG16(weights='imagenet', input_shape=input_shape, include_top=True)
    # Make all layers untrainable
    for layer in vgg_model.layers[:]:
        layer.trainable = False
    # Add fully connected layer which have 1024 neuron to VGG-16 model
    output = vgg_model.get_layer('fc2').output
    output = Flatten(name='new_flatten')(output)
    output = Dense(units=1024, activation='relu', name='new_fc')(output)
    output = Dense(units=50, activation='softmax')(output)
    vgg_model = Model(vgg_model.input, output)
    # Make last 4 layers trainable if lastFourTrainable == True
    if lastFourTrainable == True:
        vgg_model.get_layer('block5_conv3').trainable = True
        vgg_model.get_layer('fc1').trainable = True
        vgg_model.get_layer('fc2').trainable = True
        vgg_model.get_layer('new_fc').trainable = True
    # Compile VGG-16 model
    vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    vgg_model.summary()

    return vgg_model


# In[ ]:


# Get VGG-19 Model
def getVGG19Model(lastFourTrainable=False):
    vgg_model_19 = VGG19(weights='imagenet', input_shape=input_shape, include_top=True)
    # Make all layers untrainable
    for layer in vgg_model_19.layers[:]:
        layer.trainable = False
    # Add fully connected layer which have 1024 neuron to VGG-16 model
    output = vgg_model_19.get_layer('fc2').output
    output = Flatten(name='new_flatten')(output)
    output = Dense(units=1024, activation='relu', name='new_fc')(output)
    output = Dense(units=50, activation='softmax')(output)
    vgg_model_19 = Model(vgg_model_19.input, output)
    # Make last 4 layers trainable if lastFourTrainable == True
    if lastFourTrainable == True:
        vgg_model_19.get_layer('block5_conv3').trainable = True
        vgg_model_19.get_layer('fc1').trainable = True
        vgg_model_19.get_layer('fc2').trainable = True
        vgg_model_19.get_layer('new_fc').trainable = True
    # Compile VGG-16 model
    vgg_model_19.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    vgg_model_19.summary()

    return vgg_model_19


# In[ ]:


vgg_model19_a = getVGG19Model(lastFourTrainable=False)


# In[ ]:


vgg_model19_b = getVGG19Model(lastFourTrainable=True)


# In[ ]:


# Get ResNet-50 Model with lastFourTrainable=False
resnet_model_a = getResNet50Model(lastFourTrainable=False)


# In[ ]:


def score_train(model,test_x,test_y):
    # Score trained model.
    train_scores = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', train_scores[0])
    print('Test accuracy:', train_scores[1])


# In[ ]:


from tensorflow.keras.optimizers import Adam


# In[ ]:


print(validation_x_final.shape)
print(validation_y_final.shape)


# In[ ]:


import tensorflow


# In[ ]:


train_x_final_resnet=tensorflow.keras.applications.resnet.preprocess_input(train_x_final)
validation_x_final_resnet=tensorflow.keras.applications.resnet.preprocess_input(validation_x_final)


# In[ ]:


test_x_final_resnet=tensorflow.keras.applications.resnet.preprocess_input(X_test)


# In[ ]:


# Train ResNet-50 Model 
#resnet_model_a.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history1 = resnet_model_a.fit(train_x_final_resnet, train_y_final, epochs=30,validation_data=(validation_x_final_resnet, validation_y_final))
#resnet_model_a = trainModelAndGetConfusionMatrix(resnet_model_a,train_x_final,validation_x_final,X_test,10,64)


# In[ ]:


plot_hist(history1,'Resnet-50')


# In[ ]:


score_train(resnet_model_a,test_x_final_resnet,Y_test)


# In[ ]:


resnet_model_b = getResNet50Model(lastFourTrainable=True)


# In[ ]:


# Train ResNet-50 Model 
#resnet_model_a.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = resnet_model_b.fit(train_x_final_resnet, train_y_final, epochs=30,validation_data=(validation_x_final_resnet, validation_y_final))
#resnet_model_a = trainModelAndGetConfusionMatrix(resnet_model_a,train_x_final,validation_x_final,X_test,10,64)


# In[ ]:


plot_hist(history,'Resnet-50-b')


# In[ ]:


score_train(resnet_model_b,test_x_final_resnet,Y_test)


# In[ ]:


train_x_final_vgg16=tensorflow.keras.applications.vgg16.preprocess_input(train_x_final)
validation_x_final_vgg16=tensorflow.keras.applications.vgg16.preprocess_input(validation_x_final)


# In[ ]:


test_x_final_vgg16=tensorflow.keras.applications.vgg16.preprocess_input(X_test)


# In[ ]:


vgga=getVGG16Model(lastFourTrainable=False)


# In[ ]:


history = vgga.fit(train_x_final_vgg16, train_y_final,validation_data=(validation_x_final_vgg16,validation_y_final), epochs=30)


# In[ ]:


plot_hist(history,'VGG-16')


# In[ ]:


score_train(vgga,test_x_final_vgg16,Y_test)


# In[ ]:


vggb=getVGG16Model(lastFourTrainable=True)


# In[ ]:


history_vgg16_b = vggb.fit(train_x_final_vgg16, train_y_final,validation_data=(validation_x_final_vgg16,validation_y_final), epochs=30)


# In[ ]:


plot_hist(history_vgg16_b,'Vgg-16-b')


# In[ ]:


score_train(vggb,test_x_final_vgg16,Y_test)


# In[ ]:


train_x_final_vgg19=tensorflow.keras.applications.vgg19.preprocess_input(train_x_final)
validation_x_final_vgg19=tensorflow.keras.applications.vgg19.preprocess_input(validation_x_final)


# In[ ]:


test_x_final_vgg19=tensorflow.keras.applications.vgg19.preprocess_input(X_test)


# In[ ]:


history_vgg19_a = vgg_model19_a.fit(train_x_final_vgg19, train_y_final,validation_data=(validation_x_final_vgg19,validation_y_final), epochs=30)


# In[ ]:


score_train(vgg_model19_a,test_x_final_vgg19,Y_test)


# In[ ]:


plot_hist(history_vgg19_a,'vgg-19-a')


# In[ ]:


history_vgg19_b = vgg_model19_b.fit(train_x_final_vgg19, train_y_final,validation_data=(validation_x_final_vgg19,validation_y_final), epochs=30)


# In[ ]:


plot_hist(history_vgg19_b,'VGG-19-b')


# In[ ]:


score_train(vgg_model19_b,test_x_final_vgg19,Y_test)


# In[ ]:


from keras.applications.densenet import DenseNet121


# In[ ]:


# Get DenseNet-121 Model
def getDenseNet121Model(lastFourTrainable=False):
    densenet_model = DenseNet121(weights='imagenet', input_shape=input_shape, include_top=True)
    # Make all layers non-trainable
    for layer in densenet_model.layers[:]:
        layer.trainable = False
    # Add fully connected layer which have 1024 neuron to ResNet-50 model
    output = densenet_model.get_layer('avg_pool').output
    output = Flatten(name='new_flatten')(output)
    output = Dense(units=1024, activation='relu', name='new_fc')(output)
    predictions = Dense(units=50, activation='softmax')(output)
    densenet_model = Model(densenet_model.input, predictions)
    # Make last 4 layers trainable if lastFourTrainable == True
    if lastFourTrainable == True:
        densenet_model.get_layer('conv5_block3_2_bn').trainable = True
        densenet_model.get_layer('conv5_block3_3_conv').trainable = True
        densenet_model.get_layer('conv5_block3_3_bn').trainable = True
        densenet_model.get_layer('new_fc').trainable = True
    # Compile ResNet-50 model
    densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    densenet_model.summary()
    return densenet_model


# In[ ]:


# Get DenseNet-121 Model with lastFourTrainable=False
densenet_model_a = getDenseNet121Model(lastFourTrainable=False)


# In[ ]:


train_x_final_densenet=tensorflow.keras.applications.densenet.preprocess_input(train_x_final)
validation_x_final_densenet=tensorflow.keras.applications.densenet.preprocess_input(validation_x_final)


# In[ ]:


test_x_final_densenet=tensorflow.keras.applications.densenet.preprocess_input(X_test)


# In[ ]:


# Train DENSETNET-121 Model 
#resnet_model_a.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history_densenet_model_a = densenet_model_a.fit(train_x_final_densenet, train_y_final, epochs=30,validation_data=(validation_x_final_densenet, validation_y_final))
#resnet_model_a = trainModelAndGetConfusionMatrix(resnet_model_a,train_x_final,validation_x_final,X_test,10,64)


# In[ ]:


plot_hist(history_densenet_model_a,'Densenet')
score_train(densenet_model_a,test_x_final_densenet,Y_test)


# In[ ]:


from keras.models import Model


# In[ ]:


feature_model_vgg_model19_a= Model(inputs=vgg_model19_a.input, outputs=vgg_model19_a.get_layer('new_fc').output)


# In[ ]:


feature_vects=feature_model_vgg_model19_a.predict(train_x_final_vgg19)


# In[ ]:


feature_vects.shape


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


labels_integers=np.argmax(train_y_final, axis=1)


# In[ ]:


labels_integers.shape


# In[ ]:


knn = KNeighborsClassifier()
knn.fit(feature_vects,labels_integers)


# In[ ]:


test_vects=feature_model_vgg_model19_a.predict(test_x_final_vgg19)


# In[ ]:


test_vects.shape


# In[ ]:


Y_test.shape


# In[ ]:


test_label_integers=np.argmax(Y_test, axis=1)


# In[ ]:


test_label_integers.shape


# In[ ]:


print(knn.score(test_vects,test_label_integers))
#ypred=knn.predict(fin_test_img)


# In[ ]:


knn7 = KNeighborsClassifier(n_neighbors = 7)
knn7.fit(feature_vects,labels_integers)


# In[ ]:


indices=knn7.kneighbors(test_vects, return_distance=False)


# In[ ]:


indices.shape


# In[ ]:


np.max(indices)


# In[ ]:


np.min(indices)


# In[ ]:


def show_neighbors(orig, neighbors):
    f, axarr = plt.subplots(4, 2)
    for i, ax in enumerate(axarr.flatten()):
        if i == 0:
            ax.set_title("Query image")
            ax.imshow(orig)
        else:
            ax.set_title(f"Neighbor {i}")
            ax.imshow(train_x_final[neighbors[i-1]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.tight_layout()
    plt.show()


# In[ ]:


indices=knn7.kneighbors(test_vects, return_distance=False)
for i in range(0,5):
    show_neighbors(X_test[i],indices[i])


# In[ ]:


ypred=knn7.predict(test_vects)


# In[ ]:


from sklearn import metrics 
print(metrics.classification_report(test_label_integers,ypred))


# In[ ]:


get_ipython().system('pip install -q xplique')


# In[ ]:


import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import xplique
from xplique.plots import plot_attributions


# In[ ]:


print(X_train.shape)


# In[ ]:


Y_train.shape


# In[ ]:


from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod)

# to explain the logits is to explain the class, 
# to explain the softmax is to explain why this class rather than another
# it is therefore recommended to explain the logit
model = vgg_model19_a
model.layers[-1].activation = tf.keras.activations.linear
batch_size = 64

explainers = [
             Saliency(model),
             GradientInput(model),
             GuidedBackprop(model),
             IntegratedGradients(model, steps=50, batch_size=batch_size),
             SmoothGrad(model, nb_samples=50, batch_size=batch_size),
             SquareGrad(model, nb_samples=50, batch_size=batch_size),
             VarGrad(model, nb_samples=50, batch_size=batch_size),
             GradCAM(model),
             GradCAMPP(model),
             Occlusion(model, patch_size=10, patch_stride=10, batch_size=batch_size),
             SobolAttributionMethod(model, batch_size=batch_size),
             # Rise(model, nb_samples=4000, batch_size=batch_size),
             # Lime(model, nb_samples = 1000),
             # KernelShap(model, nb_samples = 1000)
]

explanations_to_test = {}

for explainer in explainers:

  explainer_name = explainer.__class__.__name__
  explanations = explainer(X_train, Y_train)

  if len(explanations.shape) > 3:
    explanations = np.mean(explanations, -1)

  # store the explanations to use the metrics
  explanations_to_test[explainer_name] = explanations
  
  print(f"Method: {explainer_name}")
  plot_attributions(explanations[:5], X_train[:5], cmap='jet', alpha=0.4,
                    cols=5, clip_percentile=0.5, absolute_value=True)
  plt.show()
  print("\n")


# In[ ]:





# In[ ]:




