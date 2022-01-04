from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from skimage.util.shape import view_as_windows
import numpy as np

class PatchesGen(Sequence):
    def __init__(self, image, labels, patch_size, overlap, tiles_size, tiles_sel, percentage, batch_size):
        self.img_shape = image.shape[0:2]
        self.labels = labels.flatten()
        self.image=image.reshape(self.img_shape[0]*self.img_shape[1],-1)
        self.patch_size = patch_size
        self.tiles_size = tiles_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.create_tiles()
        self.create_datasets(tiles_sel)
        self.create_patches(percentage)
        self.shuffle_patches()
        

    def create_tiles(self):
        tiles_x = np.array_split(np.arange(self.img_shape[0]), self.tiles_size[0])
        tiles_y = np.array_split(np.arange(self.img_shape[1]), self.tiles_size[1])

        self.tiles_array = np.zeros(self.img_shape, np.uint8)

        for tile_x in range(self.tiles_size[0]):
            for tile_y in range(self.tiles_size[1]):
                tile_num = tile_x*self.tiles_size[1]+ tile_y + 1

                self.tiles_array[tiles_x[tile_x][0]:tiles_x[tile_x][-1]+1 ,tiles_y[tile_y][0]:tiles_y[tile_y][-1]+1] = tile_num

    def create_datasets(self, tiles_sel):
        self.datasets = np.zeros(self.img_shape, dtype=bool)
        for tile in tiles_sel:
            self.datasets[self.tiles_array == tile] = 1
    
    def create_patches(self, percentage):
        row_steps, cols_steps = int((1-self.overlap) * self.patch_size), int((1-self.overlap) * self.patch_size)
        img_idx = np.arange(self.img_shape[0]*self.img_shape[1], dtype=np.uint32).reshape(self.img_shape)

        idx_patches =  view_as_windows(img_idx, (self.patch_size, self.patch_size), step=(row_steps, cols_steps)).reshape(-1,self.patch_size, self.patch_size)
        #label_patches =  view_as_windows(self.labels, (self.patch_size, self.patch_size), step=(row_steps, cols_steps)).reshape(-1,self.patch_size, self.patch_size)
        ds_patches =  view_as_windows(self.datasets, (self.patch_size, self.patch_size), step=(row_steps, cols_steps)).reshape(-1,self.patch_size, self.patch_size)

        idx_sel = np.squeeze(np.where(ds_patches.sum(axis=(1, 2))==self.patch_size**2)) 

        self.patches = idx_patches[idx_sel]

        #flat_labels = self.labels.flatten()

        remove_idx = []
        for idx, patch in enumerate(self.patches):
            label_patch = self.labels[patch]
            if (100*len(label_patch[label_patch==1])/(self.patch_size**2))<=percentage:
                remove_idx.append(idx)

        self.patches = np.delete(self.patches, remove_idx, axis=0)
        self.patches_idx = np.arange(self.patches.shape[0])

    def shuffle_patches(self):
        np.random.shuffle(self.patches_idx)
        
    def __len__(self):
        return int(self.patches.shape[0]/self.batch_size)+1

    def __getitem__(self, index):
        item_idx = self.patches_idx[index*self.batch_size:(index+1)*self.batch_size]
        ps = self.patches[item_idx]
        cs = np.random.choice([0,1], (3,ps.shape[0]))

        #random hor flip
        nhflip = np.where(cs[0]==1)[0]
        ps[nhflip] = np.flip(ps[nhflip], axis=1)

        #random vert flip
        nvflip = np.where(cs[1]==1)[0]
        ps[nvflip] = np.flip(ps[nvflip], axis=2)

        #random 
        nrot = np.where(cs[2]==1)[0]
        ps[nrot] = np.rot90(ps[nrot], axes = (1,2))

        return (self.image[ps], to_categorical(self.labels[ps]))

    def on_epoch_end(self):
        self.shuffle_patches()




