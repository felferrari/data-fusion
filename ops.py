import numpy as np
import math as m
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.models import Model
from osgeo import gdal
from tensorflow.keras.layers import Input
from skimage.morphology import area_opening
from skimage.util.shape import view_as_windows
from sklearn.metrics import confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat
from libtiff import TIFF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def load_tif_image(patch):
    # Read tiff Image
    print (patch)
    img_tif = TIFF.open(patch)
    img = img_tif.read_image()
    return img

def load_SAR_image(patch):
    '''Function to read SAR images'''
    print (patch)
    img_tif = TIFF.open(patch)
    db_img = img_tif.read_image()
    temp_db_img = 10**(db_img/10)
    temp_db_img[temp_db_img>1] = 1
    return temp_db_img

'''
def resize_image(image, height, width):
    im_resized = np.zeros((height, width, image.shape[2]), dtype='float32')
    for b in range(image.shape[2]):
        band = Image.fromarray(image[:,:,b])
        #(width, height) = (ref_2019.shape[1], ref_2019.shape[0])
        im_resized[:,:,b] = np.array(band.resize((width, height), resample=Image.NEAREST))
    return im_resized
'''

def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img

def normalization(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1

def generate_patches(img, size, stride):
    temp_image = np.pad(img, ((size, size), (size, size), (0, 0)), 'symmetric')
    overlap = int((size-stride)/2)
    patches = []
    for line in range(m.ceil(img.shape[0]/stride)):
        for col in range(m.ceil(img.shape[1]/stride)):
            l0 = size+line*stride-overlap
            c0 = size+col*stride-overlap
            patch = temp_image[l0:l0+size, c0:c0+size, :]
            patches.append(patch)

    return np.array(patches)


def generate_save_patches(img, size, stride, save_path, prefix):
    temp_image = np.pad(img, ((size, size), (size, size), (0, 0)), 'symmetric')
    overlap = int((size-stride)/2)
    i = 0
    for line in tqdm(range(m.ceil(img.shape[0]/stride))):
        for col in range(m.ceil(img.shape[1]/stride)):
            i += 1
            l0 = size+line*stride-overlap
            c0 = size+col*stride-overlap
            patch = temp_image[l0:l0+size, c0:c0+size, :]
            np.save(os.path.join(save_path, f'{prefix}_{i:07d}'), patch)



def crop_img(img, final_size):
    crop_size = int((img.shape[0] - final_size)/2)
    return img[crop_size:crop_size+final_size, crop_size:crop_size+final_size, :]

def create_idx_image(ref_mask):
    return  np.arange(ref_mask.shape[0] * ref_mask.shape[1]).reshape(ref_mask.shape[0] , ref_mask.shape[1])

def extract_patches(im_idx, patch_size, overlap):
    '''overlap range: 0 - 1 '''
    row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
    return view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))

def create_mask(size_rows, size_cols, grid_size=(6,3)):
    rows = np.array_split(np.arange(size_rows), grid_size[0])
    cols = np.array_split(np.arange(size_cols), grid_size[1])

    #num_tiles_rows = size_rows//grid_size[0]
    #num_tiles_cols = size_cols//grid_size[1]
    #print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    #patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((size_rows, size_cols), dtype=np.uint8)
    count = 0
    for row in rows:
        for col in cols:
            patch = np.ones((row.size, col.size))
            count += 1
            mask[row[0]:row[-1]+1, col[0]:col[-1]+1] = patch*count
    #plt.imshow(mask)
    #print('Mask size: ', mask.shape)
    return mask


def retrieve_idx_percentage(reference, patches_idx_set, patch_size, pertentage = 5):
    #count = 0
    new_idx_patches = []
    reference_vec = reference.reshape(reference.shape[0]*reference.shape[1])
    for patchs_idx in patches_idx_set:
        patch_ref = reference_vec[patchs_idx]
        class1 = patch_ref[patch_ref==1]
        if len(class1) >= int((patch_size**2)*(pertentage/100)):
            #count = count + 1
            new_idx_patches.append(patchs_idx)
    return np.asarray(new_idx_patches)

'''
Load the Optical Imagery -img-. Usually GDAL opens the image in [layers, height and width] order and need to be changed 
to [height, width and layers] order.
'''

def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed

def load_opt(img):
    return np.moveaxis(gdal.Open(img).ReadAsArray(), 0, 2)


def load_sar(img):
    temp = np.expand_dims(gdal.Open(img).ReadAsArray(), axis=-1)
    temp = 10**(temp/10)
    temp[temp > 1] = 1
    return temp


def min_max_scaler(img):
    scaler = MinMaxScaler()
    shape = img.shape
    return scaler.fit_transform(np.expand_dims(img.flatten(), axis=-1)).reshape(shape)

def summary(layer, inputs):
    x = [Input(shape=inp) for inp in inputs]
    model = Model(x, layer.call(x))
    return model.summary()


def metric_thresholds(thr, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    print(thr)
    img_reconstructed = np.zeros_like(prob_map, dtype=np.uint8)
    
    img_reconstructed[prob_map >= thr] = 1

    mask_areas_pred = np.ones_like(ref_reconstructed, dtype=np.uint8)
    area = np.uint8(area_opening(img_reconstructed, area_threshold = px_area, connectivity=1))
    area_no_consider = np.uint8(img_reconstructed-area)
    mask_areas_pred[area_no_consider==1] = 0
    
    # Mask areas no considered reference
    mask_borders = np.ones_like(img_reconstructed, dtype=np.uint8)
    #ref_no_consid = np.zeros((ref_reconstructed.shape))
    mask_borders[ref_reconstructed==2] = 0
    #mask_borders[ref_reconstructed==-1] = 0
    
    mask_no_consider = np.uint8(mask_areas_pred * mask_borders)
    ref_consider = np.uint8(mask_no_consider*ref_reconstructed)
    pred_consider = np.uint8(mask_no_consider*img_reconstructed)
    
    ref_final = ref_consider[mask_amazon_ts_==1]
    pre_final = pred_consider[mask_amazon_ts_==1]
    
    # Metrics
    cm = confusion_matrix(ref_final, pre_final)

    #TN = cm[0,0]
    FN = cm[1,0]
    TP = cm[1,1]
    FP = cm[0,1]
    precision_ = TP/(TP+FP)
    recall_ = TP/(TP+FN)
    aa = (TP+FP)/len(ref_final)
    mm = np.hstack((recall_, precision_, aa))
    return mm

def metrics_AP(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area, processes = 1):
    if processes > 1:
        pool = Pool(processes=processes)
        metrics = pool.starmap(
            metric_thresholds, 
            zip(
                thresholds_, 
                repeat(prob_map),
                repeat(ref_reconstructed),
                repeat(mask_amazon_ts_),
                repeat(px_area),
                )
            )
        return metrics
    else:
        metrics = []
        for thr in thresholds_:
            metrics.append(metric_thresholds(thr, prob_map, ref_reconstructed, mask_amazon_ts_, px_area))
            
        return metrics

       

def complete_nan_values(metrics):
    vec_prec = metrics[:,1]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = 2*vec_prec[j+1]-vec_prec[j+2]
            if vec_prec[j] >= 1:
                vec_prec[j] == 1
    metrics[:,1] = vec_prec
    return metrics 


