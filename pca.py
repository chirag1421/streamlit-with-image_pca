#load libraries

import glob
import xlsxwriter as xlsxwriter
from PIL import Image
import numpy as np
import pandas as pd
import os
import streamlit as st
import seaborn as sns
# rename the database file images

def rename_cardata():
  for count,filename in enumerate(os.listdir("/Users/chirag/PycharmProjects/carimage_pca/cardata")):
    dst = "car1" + str(count) + ".png"
    src = "/Users/chirag/PycharmProjects/carimage_pca/cardata" + filename
    dst = "/Users/chirag/PycharmProjects/carimage_pca/cardata" + dst
    os.rename(src,dst)
  rename_cardata()

# process of the gray scaled image

path = "/Users/chirag/PycharmProjects/carimage_pca/cardata/*.*"
NEW_SIZE = (250, 250)
i = 0
for file in glob.glob(path):
    img = Image.open(file)
    st.image(img)
    print("car dataset", i, ":")
    print("Original Size:", end="")
    print(img.size, end="")
    imgGray = img.convert('L')
    print("Resize:", end="")
    img1 = imgGray.resize(NEW_SIZE)
    print(img1.size)
    st.success(img1.save(f'/Users/chirag/PycharmProjects/carimage_pca/GrayScale_Resized/cardata_image_{i}.png'))
    i += 1

# display gray scaled image

img1 = Image.open("/Users/chirag/PycharmProjects/carimage_pca/GrayScale_Resized/cardata_image_1.png")
st.image(img1)

#extracting pixel values
workbook = xlsxwriter.Workbook('/Users/chirag/PycharmProjects/carimage_pca/data.xlsx')
worksheet = workbook.add_worksheet()
row=0
column = 0



#extracting the pixle values and stored values

path = "/Users/chirag/PycharmProjects/carimage_pca/GrayScale_Resized/*.*"
for file in glob.glob(path):
  img = Image.open(file)
  pixels = np.array(img)
  pix = pixels.flatten()
  pix_lst = pix.tolist()
  worksheet.write_row(row,column,pix_lst)
  row += 1
  print(pix)
workbook.close()


df = pd.read_excel('/Users/chirag/PycharmProjects/carimage_pca/data.xlsx',header=None)
col = [*range(0,16384,1)]
features = ['X'+str(a) for a in col]
df = df.set_axis(features, axis=1, copy=False)
df.loc['y']=0
st.write(df.head())
st.write(df.shape)

import cv2
import zipfile
import numpy as np

cars = {}
with zipfile.ZipFile("/Users/chirag/PycharmProjects/carimage_pca/GrayScale_Resized.zip") as carzip:
    for filename in carzip.namelist():
        if not filename.endswith(".png"):
            continue  # not a face picture
        with carzip.open(filename) as image:
            # If we extracted files from zip, we can use cv2.imread(filename) instead
            cars[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
carimages = list(cars.values())[-16:]  # take last 16 images
for i in range(16):
    axes[i % 4][i // 4].imshow(carimages[i], cmap="gray")
st.pyplot(fig)

carshape = list(cars.values())[0].shape
st.write("Car image shape:", carshape)

st.write(list(cars.keys())[:5])

classes = set(cars.keys())
st.write("Number of pictures:", len(cars))

# Take classes 1-1062 for eigenfaces, keep entire class 40 and
# image 10 of class 1062 as out-of-sample test
carmatrix = []
carlabel = []
for key, val in cars.items():
    if key.startswith("/content/drive/MyDrive/car/images/dataset/GrayScale_Resized/"):
        continue  # this is our test set
    if key == "/content/drive/MyDrive/car/images/dataset/GrayScale_Resized/cardata_image_1.png":
        continue  # this is our test set
    carmatrix.append(val.flatten())
    carlabel.append(key[0])

# Create carmatrix as (n_samples,n_pixels) matrix
carmatrix = np.array(carmatrix)

# Apply PCA to extract eigencars
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA().fit(carmatrix)

st.write(pca.explained_variance_ratio_)

# Take the first K principal components as eigenfaces
n_components = 50
eigencars = pca.components_[:n_components]

# Show the first 16 eigenfaces
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i // 4].imshow(eigencars[i].reshape(carshape), cmap="gray")
st.pyplot(fig)

# Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
weights = eigencars @ (carmatrix - pca.mean_).T

weights = []

for i in range(carmatrix.shape[0]):
    weight = []
    for j in range(n_components):
        w = eigencars[j] @ (carmatrix[i] - pca.mean_)
        weight.append(w)
    weights.append(weight)

# Test on out-of-sample image of existing class
query = cars["cardata_image_0.png"].reshape(1,-1)
query_weight = eigencars @ (query - pca.mean_).T
euclidean_distance = np.linalg.norm(weight - query_weight, axis=0)
best_match = np.argmin(euclidean_distance)
st.write("Best match %s with Euclidean distance %f" % (carlabel[best_match], euclidean_distance[best_match]))

# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(carshape), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(carmatrix[best_match].reshape(carshape), cmap="gray")
axes[1].set_title("Best match")
st.pyplot(fig)

