#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


# Function to calculate the RMS Deviation from a 5x5 grid around a central pixel
# Input 1 is the RGB-IR Aerial Image Raster
# Input 2 is the Classfication Results from the NN where green trees and grass are combined to 4.
# Input 3 is the classification for green trees+grass
def RMS_of_band5x5(band, image_predict, classification):
    
    #Padding the image
    prediction_matrix = np.zeros((len(band)+4, len(band[0])+4))
    matrix = np.zeros((len(band)+4, len(band[0])+4))
    matrix[2:len(band)+2,2:len(band[0])+2]=band
    prediction_matrix[2:len(band)+2,2:len(band[0])+2]  = image_predict
    prediction_matrix = prediction_matrix.astype('int')
    
    #Initializing RMS Matrix
    RMS_matrix=np.zeros(band.shape)
    
    for i in range(len(band)):
        for j in range(len(band[0])):
            if image_predict[i,j]==classification:
                image_chip = matrix[i:i+5,j:j+5].flatten()
                image_chip_pred = prediction_matrix[i:i+5,j:j+5].flatten()
                
                calc_image=image_chip[image_chip_pred==classification]
                RMS=calc_image.std()
                
                RMS_matrix[i,j]=RMS
            else:
                RMS_matrix[i,j]=-1
    return RMS_matrix


# In[5]:


#function to run the classification once the NN is trained.
#input is the AerialImage raster (tif) with RGB-IR data
#output is the Classification raster (tif) with 1-8 for each fuel type.
def classify(inputfile, outputfile):
    #loading raster
    image_raster = rio.open(inputfile)
    image_data = image_raster.read()
    
    #Getting Coordinate Meshgrid
    coordinate_data_x = np.zeros((image_raster.height, image_raster.width))
    coordinate_data_y = np.zeros((image_raster.height, image_raster.width))

    for i in range(image_raster.height):
        for j in range(image_raster.width):
            coords = image_raster.xy(i, j)
            coordinate_data_x[i,j]=coords[0]
            coordinate_data_y[i,j]=coords[1]
    
    #Creating Machine Learning Data Set
    band0 = image_data[0].flatten()
    band1 = image_data[1].flatten()
    band2 = image_data[2].flatten()
    band3 = image_data[3].flatten()
    
    features_data = np.array([band0,band1,band2,band3])
    X = features_data.T
    X_cut = X[band0!=0]
    X_scaled = scaler.transform(X_cut)
    
    #Prediction using Neural Network
    y_predict_cut = mlp.predict(X_scaled)
    y_predict = 999*np.ones(len(X))
    y_predict[band0!=0]=y_predict_cut
    image_predict = np.reshape(y_predict, image_data[0].shape)
    
    #RMS Method
    NIR_RMS = RMS_of_band5x5(image_data[3], image_predict, 4)
    
    NIR_RMS_f = NIR_RMS.flatten()
    
    for i in range(len(NIR_RMS_f)):
        if NIR_RMS_f[i]<15.472021 and y_predict[i]==4:
            y_predict[i]=6
        
    image_predict = np.reshape(y_predict, image_data[0].shape)
    
    
    #Saving as tif raster file
    xmin,ymin,xmax,ymax = [coordinate_data_x.min(),
                           coordinate_data_y.min(),
                           coordinate_data_x.max(),
                           coordinate_data_y.max()]
    nrows,ncols = image_predict.shape
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0,-yres)   

    output_raster = gdal.GetDriverByName('GTiff').Create(outputfile,
                                                         ncols, nrows, 1,
                                                         gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)  
    srs = osr.SpatialReference()                 
    srs.ImportFromEPSG(32611)                    
                                             
    output_raster.SetProjection( srs.ExportToWkt() )   
    output_raster.GetRasterBand(1).WriteArray(image_predict)

    output_raster.FlushCache()
    
    print("done.")
    
    return image_predict


# In[7]:


#TRAINING THE NEURAL NETWORK

#Loading the raster
image_raster = rio.open(r'hxip_m_3711955_nw_11_100.tif')

#Loading the data read() returns array with shape (bands, rows, columns)
image_data = image_raster.read()

#Loading the coordinate data
#Returns a mesh grid, unknown units
coordinate_data_x = np.zeros((image_raster.height, image_raster.width))
coordinate_data_y = np.zeros((image_raster.height, image_raster.width))

for i in range(image_raster.height):
    for j in range(image_raster.width):
        coords = image_raster.xy(i, j)
        coordinate_data_x[i,j]=coords[0]
        coordinate_data_y[i,j]=coords[1]
        
#Loading classification data
classification_raster = rio.open(r'hxip_m_3711955_nw_11_100_tag_04252022B.tif')
classification = classification_raster.read()

coordinate_data_x_classification = np.zeros((classification_raster.height, classification_raster.width))
coordinate_data_y_classification = np.zeros((classification_raster.height, classification_raster.width))

for i in range(classification_raster.height):
    for j in range(classification_raster.width):
        coords = classification_raster.xy(i, j)
        coordinate_data_x_classification[i,j]=coords[0]
        coordinate_data_y_classification[i,j]=coords[1]
        
#Machine Learning "Features" Data set
#[band0, band1, band2, band3, x, y]

band0 = image_data[0].flatten()
band1 = image_data[1].flatten()
band2 = image_data[2].flatten()
band3 = image_data[3].flatten()

features_data = np.array([band0,band1,band2,band3])
X0 = features_data.T
image = X0

#Tagged Data
y0 = classification[0].flatten()
y0 = y0.astype(int)

#Cuts (Train/Test Split):
X_train = X0[y0!=0]
y = y0[y0!=0]

for i in range(len(y)):
    if y[i]==6:
        y[i]=4


# In[10]:


#Scaling of All Data Sets
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#For Supervised Learning
X_train_scaled = scaler.fit_transform(X_train)

from sklearn.neural_network import MLPClassifier

#Predicting whole data set

mlp = MLPClassifier(solver = 'sgd', activation = 'tanh', 
                    random_state = 0, hidden_layer_sizes = [15,15,15,15])
mlp.fit(X_train_scaled, y)

print("Done Training.")


# In[ ]:


predictionNW = classify(r'hxip_m_3711955_nw_11_100.tif', 
                        r'C:\Users\wto\OneDrive - Stan State\Research\Wildfire\Fuelmap\Wing\NNRMS_3711955_nw_11_100.tif')


# In[ ]:




