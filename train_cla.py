import datetime
from gc import callbacks

from loaders.data_loader import Data_Seq
from models.pointnet_cla import Pointnet_Cla

import tensorflow as tf

if __name__ == "__main__":

    num_point = 128
    batch_size = 32
    epochs = 20
    ite_size = 10
    train_seq = Data_Seq("./dataset/trimesh_primitives/train", num_point, batch_size, ite_size)
    val_seq = Data_Seq("./dataset/trimesh_primitives/val", num_point, batch_size, 1)

    pointnet_cla = Pointnet_Cla(num_point, 4)
    pointnet_cla.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    pointnet_cla.fit(
        x=train_seq,
        batch_size=batch_size,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./logs/weights-{}.h5".format(ts)
    print("save weights as :", save_path)
    pointnet_cla.save_weights(save_path)