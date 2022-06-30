from dataclasses import replace
import sys 
import os
import numpy as np
import trimesh
import tensorflow as tf

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_dir)

def create_contents_list(dataPath):
    t = os.listdir(dataPath)
    t = [os.path.join(dataPath, c) for c in t]
    files = [c for c in t if os.path.isfile(c)]
    directories = [c for c in t if os.path.isdir(c)]
    return files, directories

class Data_Seq(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, num_points, batch_size, iter_size):
        self.dataset_dir = dataset_dir
        self.num_points = num_points
        self.batch_size = batch_size
        self.iter_size = iter_size
        
        self.data_path, self.data_label = self.load_data()
        print("data_path:" , self.data_path[:3], " ... ", self.data_path[-3:])
        print("data_label:" , self.data_label[:3], " ... ", self.data_label[-3:])

    def load_data(self):
        data_path, data_label = list(), list()
        _, directories = create_contents_list(self.dataset_dir)
        print("directories: ", directories)
        for l, d in enumerate(directories):
            files, _ = create_contents_list(d)
            file_num = len(files)
            data_path += files
            data_label += [l] * file_num
        return data_path, data_label

    def getitem(self):
        """
        for test
        returns x, y data of the batch size.
        """
        return self.__getitem__(0)

    def __getitem__(self, idx):
        batch_x = list()
        batch_y = list()
        ### use single thread
        for i in range(self.batch_size):
            x, y, _ = self.Preprocess(i)
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x).astype(np.float32)
        batch_y = np.array(batch_y).astype(np.float32)
        batch_y = np.expand_dims(batch_y, -1)
        return batch_x, batch_y

    def __len__(self):
        return self.batch_size * self.iter_size

    def Preprocess(self, i):
        rind = np.random.randint(len(self.data_path))
        data_path = self.data_path[rind]
        y = self.data_label[rind]
        x = trimesh.load(data_path)
        ## augment data here as appropriate
        x = x.vertices
        samples_id = np.random.choice(np.arange(x.shape[0]), self.num_points, replace=False)
        x = x[samples_id]
        return x, y, i 

if __name__ == "__main__":
    data_seq = Data_Seq("./dataset/trimesh_primitives/train", 128, 32, 10)
    x, y = data_seq.getitem()
    print("x.shape: ", x.shape)
    print("x[0]: ", x[0])

    print("y.shape: ", y.shape)
    print("y[:10]: ", y[:10])


    
    