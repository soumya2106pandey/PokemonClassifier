import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
# import seaborn as snsz
from mlxtend.plotting import plot_confusion_matrix
import glob



# import cv2
img = cv2.imread('D:/7th Sem/Building Innovative System/Image-classification-Using-ML-main/Butterfree/0d6b68356f4a474c87b97d791f683309.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)

scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


################## Resizing all images ##################


desired_size = 368
im_pth = "D:/7th Sem/Building Innovative System/Image-classification-Using-ML-main/Pikachu/a.jpg"

im = cv2.imread(im_pth)
old_size = im.shape[:2] # old_size is in (height, width) format

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])

# new_size should be in (width, height) format

im = cv2.resize(im, (new_size[1], new_size[0]))

delta_w = desired_size - new_size[1]
delta_h = desired_size - new_size[0]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)

color = [0, 0, 0]
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

cv2.imshow("image", new_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('D:/7th Sem/Building Innovative System/Image-classification-Using-ML-main/Pikachu/a.jpg', new_im) 


########### Reading multiple images at once

import glob

images = [cv2.imread(file) for file in glob.glob("D:\\7th Sem\\Building Innovative System\\Image-classification-Using-ML-main\\Pikachu/*.jpg")]

images_1 = [cv2.imread(file) for file in glob.glob("D:\\7th Sem\\Building Innovative System\\Image-classification-Using-ML-main\\Butterfree/*.jpg")]

images_2 = [cv2.imread(file) for file in glob.glob("D:\\7th Sem\\Building Innovative System\\Image-classification-Using-ML-main\\Ditto/*.jpg")]



###############################################################

mera_dat = []

for i in range(199):
    desired_size = 368
    
    im = images[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/rajpu/Desktop/a.jpg'.format(i), new_im) 
    mera_dat.append(new_im)


# cv2.imshow("", mera_dat[120])


############# Butterfree

mera_dat_1 = []

for i in range(66):
    desired_size = 368
    
    im = images_1[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/rajpu/Desktop/a.jpg'.format(i), new_im) 
    mera_dat_1.append(new_im)


########### Ditto
    

mera_dat_2 = []

for i in range(56):
    desired_size = 368
    
    im = images_2[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/rajpu/Desktop/a.jpg'.format(i), new_im) 
    mera_dat_2.append(new_im)


arr = np.array(mera_dat)
arr = arr.reshape((199, 406272))

ar1 = np.array(mera_dat_1)
ar1 = ar1.reshape((66, 406272))

ar2 = np.array(mera_dat_2)
ar2 = ar2.reshape((56, 406272))

arr = arr / 255
ar1 = ar1 / 255
ar2 = ar2 / 255

dataset = pd.DataFrame(arr)
dataset['label'] = np.ones(199)

dataset.iloc[:, -1]

dataset_1 = pd.DataFrame(ar1)
dataset_1['label'] = np.zeros(66)

dataset_1.iloc[:, -1]

dataset_2 = pd.DataFrame(ar2)
dataset_2['label'] = np.array(np.ones(56) + np.ones(56))

dataset_2.iloc[:, -1]

dataset_master = pd.concat([dataset, dataset_1, dataset_2])

dataset_master.iloc[:, 406272]

X = dataset_master.iloc[:, 0:406272].values
y = dataset_master.iloc[:, -1].values


fname= 'temp.csv'
dataset_master.to_csv(fname)


# decission Tree
# from sklearn.tree import DecisionTreeClassifier
# dtf = DecisionTreeClassifier(max_depth = 3)
# dtf.fit(X, y)
# dtf.score(X, y)

# y_pred1= dtf.predict(X)
from sklearn import metrics
# cm1  = metrics.confusion_matrix(y, y_pred1)
# ax= plt.axes()
# #sns.heatmap(data = cm1, annot=cm1)
# lab = ['Butter Free', 'Pikachu', 'Ditto']
# fig, ax= plot_confusion_matrix(conf_mat= cm1, show_absolute= True, show_normed= True)
# #fig, ax= plot_confusion_matrix(conf_mat= cm1, show_absolute= True, show_normed= True,class_names=lab)
# ax.set_title('Confusion Matrix for Decision Tree', fontsize= 18)
# plt.show()

# from yellowbrick.classifier import ClassificationReport

# visualizer = ClassificationReport(dtf, support=True)
# visualizer.fit(X, y)       
# visualizer.score(X, y)        
# visualizer.show() 



# # Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# nb.fit(X, y)
# nb.score(X, y)

# y_pred2= nb.predict(X)
# cm2= metrics.confusion_matrix(y,y_pred2)
# ax= plt.axes()
# fig, ax= plot_confusion_matrix(conf_mat= cm2, show_absolute= True, show_normed= True)
# ax.set_title('Confusion Matrix for Naive Bayes', fontsize= 18)
# plt.show()

# from yellowbrick.classifier import ClassificationReport

# visualizer = ClassificationReport(nb, support=True)
# visualizer.fit(X, y)       
# visualizer.score(X, y)        
# visualizer.show() 



# # Logistic Regression
# from sklearn.linear_model import LogisticRegression
# log_reg = LogisticRegression()
# log_reg.fit(X, y)
# log_reg.score(X, y)

# y_pred3= log_reg.predict(X)
# cm3= metrics.confusion_matrix(y,y_pred3)
# ax= plt.axes()
# fig, ax= plot_confusion_matrix(conf_mat= cm3, show_absolute= True, show_normed= True)
# ax.set_title('Confusion Matrix for Logistic regression', fontsize= 18)
# plt.show()

# from yellowbrick.classifier import ClassificationReport

# visualizer = ClassificationReport(log_reg, support=True)
# visualizer.fit(X, y)       
# visualizer.score(X, y)        
# visualizer.show() 


# SUpport Vector Classifier
from sklearn.svm import SVC
svm = SVC()
svm.fit(X, y)
svm.score(X, y)

y_pred4= svm.predict(X)
cm4= metrics.confusion_matrix(y,y_pred4)
ax= plt.axes()
fig, ax= plot_confusion_matrix(conf_mat= cm4, show_absolute= True, show_normed= True)
ax.set_title('Confusion Matrix for Support Vector Machine', fontsize= 18)
plt.show()

from yellowbrick.classifier import ClassificationReport

visualizer = ClassificationReport(svm, support=True)
visualizer.fit(X, y)       
visualizer.score(X, y)        
visualizer.show() 


# from sklearn.ensemble import RandomForestClassifier
# rm = RandomForestClassifier(min_samples_split=5)
# rm.fit(X,y)
# rm.score(X,y)

# y_pred5= rm.predict(X)
# cm5= metrics.confusion_matrix(y,y_pred5)
# ax= plt.axes()
# fig, ax= plot_confusion_matrix(conf_mat= cm5, show_absolute= True, show_normed= True)
# ax.set_title('Confusion Matrix for Random Forest', fontsize= 18)
# plt.show()

# visualizer = ClassificationReport(rm, support=True)
# visualizer.fit(X, y)       
# visualizer.score(X, y)        
# visualizer.show() 


# from sklearn.cluster import KMeans

# wcv = []

# for i in range(1, 8):
#     km = KMeans(n_clusters = i)
#     km.fit(X)
#     wcv.append(km.inertia_)

# plt.plot(range(1, 8), wcv)
# plt.show()


# kmeans = KMeans(n_clusters=2, init ='k-means++', max_iter=300, n_init=10,random_state=0 )

# y_kmeans = kmeans.fit_predict(X)
# plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
# #plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
# #plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
# #plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
# #plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')

# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
# plt.title('Clusters')
# plt.show()

# import tensorflow as tf
# from tensorflow import keras

# model = keras.models.Sequential()
# model.add(keras.layers.Dense(256, activation = 'relu'))
# model.add(keras.layers.Dense(128, activation = 'relu'))
# model.add(keras.layers.Dense(3, activation = 'softmax'))

# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# history = model.fit(X, y, epochs = 10)

# pd.DataFrame(history.history).plot(figsize = (8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()


# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier()
# knn.fit(X,y)
# knn.score(X,y)

# y_pred6 = knn.predict(X)
# cm6= metrics.confusion_matrix(y,y_pred6)
# ax= plt.axes()
# fig, ax= plot_confusion_matrix(conf_mat= cm6, show_absolute= True, show_normed= True)
# ax.set_title('Confusion Matrix for Random Forest', fontsize= 18)
# plt.show()

# visualizer = ClassificationReport(knn, support=True)
# visualizer.fit(X, y)       
# visualizer.score(X, y)        
# visualizer.show() 