from wrapper import Wrap
from data_op import DataManager
import pandas as pd
from my_callbacks import MyCallbacks
# from tensorflow.keras.models import save_model
dat = DataManager('hourly', 3, 6)
obj = Wrap(dat)
grid_history = obj.random_method()

# mod = grid_history.estimator.fit(x=dat.x_train, y=dat.y_train).model
# print(grid_history.estimator.__class__)
# mod.save("a.h5")


# mod = grid_history.best_estimator_.fit(x=dat.x_train, y=dat.y_train)  # this is a history obj now
# model1 = mod.model
# model1.save("b.h5")


# print(grid_history.best_estimator_.__class__ is grid_history.estimator.__class__)

best_params = grid_history.best_params_
print(best_params)

callbacks = MyCallbacks(5)
es = callbacks.early()
tb = callbacks.tensor_board()
# print(dat.x_train.shape, dat.y_train.shape)
model_x = obj.keras_regressor.build_fn().fit(x=dat.x_train, y=dat.y_train,
                                   validation_data=(dat.y_validation, dat.y_validation),
                                   batch_size=best_params['batch_size'],
                                   epochs=best_params['epochs'], verbose=1, callbacks=[es, tb])

model_x.model.save('xxx.h5')

hist = pd.DataFrame(model_x.history)
hist.to_csv('history.csv')
