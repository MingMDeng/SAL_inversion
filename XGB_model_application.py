import numpy as np
import os
from osgeo import gdal, gdalconst, ogr
import glob
import pickle
import xgboost as xgb
import cv2 as cv
import re
import pandas as pd

band_indices = [1, 2, 3, 4, 5, 6, 8]
excel_file = r"D:\iDATA\INML_v20241025\WLS20241025_pred\WLS_lakearea.xlsx"
df = pd.read_excel(excel_file)
date_values = dict(zip(df['date'].astype(str), df['value']))

def Calculate_FUI_alpha(Rrs_443, Rrs_490, Rrs_560, Rrs_665, Rrs_705):
    X = 11.756 * Rrs_443 + 6.423 * Rrs_490 + 53.696 * Rrs_560 + 32.028 * Rrs_665 + 0.529 * Rrs_705
    Y = 1.744 * Rrs_443 + 22.289 * Rrs_490 + 65.702 * Rrs_560 + 16.808 * Rrs_665 + 0.192 * Rrs_705
    Z = 62.696 * Rrs_443 + 31.101 * Rrs_490 + 1.778 * Rrs_560 + 0.015 * Rrs_665 + 0.000 * Rrs_705

    SUM= X + Y + Z
    x = X / SUM
    y = Y / SUM

    a = (np.arctan2(y - 1/3, x - 1/3) % (2 * np.pi)) * 180 / np.pi

    delta = -65.74 * (x / 100) ** 5 + 477.16 * (a / 100) ** 4 - 1279.99 * (a / 100) ** 3 + 1524.96 * (a / 100) ** 2 - 751.59 * (a / 100) + 116.56
    alpha = a + delta
    return alpha

def extract_date_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r"\d{8}", basename)
    if match:
        date = match.group()
        return date
    else:
        return None

folder_path = r"F:\Data\INML\WLS_batch_Rrs\WLS_batch"
image_files = [file for file in glob.glob(folder_path + "/*.tif*") if file.lower().endswith(('.tif', '.TIFF'))and "_Rrs" in file]  

def read_raster_bands_as_features(image_path, band_indices):
    try:
        
        dataset = gdal.Open(image_path,gdal.GA_ReadOnly)
        
        date = extract_date_from_filename(image_path)
        if date in date_values:
            value = date_values[date]
        else:
            value = None 
        print(value)
        
        width   = dataset.RasterXSize   
        height  = dataset.RasterYSize   

        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # metadata = dataset.GetMetadata()

        num_bands = dataset.RasterCount

        bands = [dataset.GetRasterBand(i).ReadAsArray() for i in band_indices]

        dataset = None

        features = np.stack(bands, axis=-1)

        NDWI = (features[:, :,2] -  features[:, :,6]) / (features[:, :,2] +  features[:, :,6])
        SI1 = features[:, :,3] / (features[:, :,3] + features[:, :,2])                            #  SI1B4/(B4+B3)
        SI2 = features[:, :,3] / (features[:, :,1] + features[:, :,2])                            #  SI2B4/(B2+B3)
        SI6 = np.divide(features[:, :, 3], features[:, :, 1], out=np.full_like(features[:, :, 3], np.nan), where=(features[:, :, 1] != 0))    

        Area = np.full((height, width), value)
        
        alpha = Calculate_FUI_alpha(features[:, :,0], features[:, :,1], features[:, :,2], features[:, :,3],features[:, :,4])

        select_features = np.dstack((features, NDWI,SI1,SI2,SI6,Area,alpha))

        return select_features, width, height, projection, geotransform, num_bands # ,metadata
    except Exception as e:
        print(f"readError image: {image_path}")
        print(f"Error message: {str(e)}")
        return None, None, None, None, None, None


for image_path in image_files:

    select_features, width, height, projection, geotransform, num_bands = read_raster_bands_as_features(image_path, band_indices)
    print("SHAPE:", select_features.shape)
    print("Row:", height)
    print("Cloum:", width)
    print("Bands:", num_bands)
    print("Proj:", projection)
    print("Geo:", geotransform)
    
    select_features_2d = np.reshape(select_features, (height * width, 13))
    
    dtrain = xgb.DMatrix(select_features_2d)
    
    model = pickle.load(open("xgb_INMLac20241025.dat", "rb")) 
    print("Loaded model from: xgb_SAL_model")
    
    feature_names = ['B1','B2','B3','B4','B5', 'B6','B8','NDWI', 'SI1B4/(B4+B3)', 'SI2B4/(B2+B3)', 'SI6(B4/B2)','Area','alpha']
    dtrain = xgb.DMatrix(select_features_2d, feature_names=feature_names)

    ypreds = model.predict(dtrain)
    ypreds = np.power(10,ypreds)
    ypreds_2d = np.reshape(ypreds, (height, width))
    
    file_name = os.path.basename(image_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    output_folder = r'D:\iDATA\INML_v20241025\WLS20241025_pred\test' 
    output_filename = file_name_without_ext+ "_pred.tif"  
    output_path = os.path.join(output_folder, output_filename)

    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    
    if output_dataset is None:
        print("outputimagefail")
    else:
        band = output_dataset.GetRasterBand(1)

        band.WriteArray(ypreds_2d)

        output_dataset.SetGeoTransform(geotransform)  
        output_dataset.SetProjection(projection) 

        output_dataset = None

        print("ouptimageDone:", output_path)