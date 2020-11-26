from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger
import time

class MyCallbacks:
    def __init__(self, patience):
        self.patience = patience
        self.es = self.early()
        self.tb = self.tensor_board()
        self.mc = self.model_checkpoint()
        self.csv_log = self.csv_logger()

    def early(self):
        my_early = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1, mode='min')
        return my_early

    def tensor_board(self):
        tensorboard = TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0,
                                  write_graph=True, write_images=False)
        return tensorboard

    def model_checkpoint(self):
        checkpoint_path = "training_1/{epoch:04d}_cp.ckpt"
        # mc = ModelCheckpoint(filepath=f'{time.time()}.h5', monitor='val_loss', mode='min', verbose=0)
        mc = ModelCheckpoint(filepath=f"{time.time()}_{checkpoint_path}", monitor='val_loss', mode='min', verbose=1, period=10)
        return mc

    def csv_logger(self):
        csv_log = CSVLogger('logs.csv')
        return csv_log
