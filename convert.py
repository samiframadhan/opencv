# Convert npz to yaml using cv2.FileStorage
import cv2 as cv
import numpy as np
import yaml

def npz_to_yaml(npz_file, yaml_file):
    data = np.load(npz_file)
    fs = cv.FileStorage(yaml_file, cv.FILE_STORAGE_WRITE)
    for key in data.files:
        fs.write(key, data[key])
    fs.release()
        
# Example usage
npz_to_yaml('calib_data/MultiMatrix.npz', 'camera_parameters2.yaml')