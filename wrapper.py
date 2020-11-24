from network_architecture import NeuralArch as Net
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from my_callbacks import MyCallbacks

class Wrap:
    """use GridSearchCV, RandomizedSearchCV and Evolutionary Search with this class.

    Methods
    -------
    grid_method(self)
        uses the GridSearchCV object and .fit()
        :returns grid_history = grid.fit()

    random_method(self)
        uses the RandomizedSearchCV object and .fit()
        :returns rand_history = rand.fit()

    Attributes
    ----------
    data_obj : (DataManager)
        passed in object from DataManager class
    network_obj: (NeuralArch)
        architecture of the neural network, so we can use it in KerasRegressor
    keras_regressor: (KerasRegressor)
        a KerasRegressor object with build_fn=network_obj.build_nn"""
    
    def __init__(self, data_obj, patience=10):
        # todo explain why do we have these different object in the class
        self.data_obj = data_obj
        network_obj = Net(self.data_obj)
        self.keras_regressor = KerasRegressor(build_fn=network_obj.build_nn)
        self.callback = MyCallbacks(patience=patience)

    def grid_method(self):
        """grid_method(self)
            uses the GridSearchCV object and .fit()
            :returns grid_history = grid.fit()"""
        
        params = dict(epochs=[200], batch_size=[4, 8])
        cv = [(slice(None), slice(None))]  # why have i written this over and over??

        es = self.callback.es
        mc = self.callback.mc
        tb = self.callback.tb
        csv_log = self.callback.csv_log
        my_callbacks = [es, mc, csv_log]

        self.grid = GridSearchCV(estimator=self.keras_regressor, param_grid=params, cv=cv)
        grid_history = self.grid.fit(X=self.data_obj.x_train, y=self.data_obj.y_train,
                                     validation_data=(self.data_obj.x_validation, self.data_obj.y_validation),
                                     verbose=0, callbacks=my_callbacks)
        return grid_history

    def random_method(self):
        """grid_method(self)
            uses the GridSearchCV object and .fit()
            returns grid_history = grid.fit()"""
        
        params = dict(epochs=[100], batch_size=[2, 4, 8, 12, 16, 20, 24, 32, 36])
        cv = [(slice(None), slice(None))]

        es = self.callback.es
        mc = self.callback.mc
        tb = self.callback.tb
        my_callbacks = [es, mc]

        self.rand = RandomizedSearchCV(estimator=self.keras_regressor, param_distributions=params, n_iter=8)
        rand_history = self.rand.fit(X=self.data_obj.x_train, y=self.data_obj.y_train,
                      validation_data=(self.data_obj.x_validation, self.data_obj.y_validation),
                      verbose=0, callbacks=my_callbacks)

        return rand_history

    def evolution_method(self):
        # this does not work, but we need to continue
        params = dict(epochs=[200], batch_size=[4, 8])
        # cv = [(slice(None), slice(None))]

        es = self.callback.es
        mc = self.callback.mc
        tb = self.callback.tb
        my_callbacks = [es, mc]

        fit_params = {"epochs": 300,
                      "validation_data": (self.data_obj.x_validation, self.data_obj.y_validation),
                      "callbacks": my_callbacks}

        self.evo = EvolutionaryAlgorithmSearchCV(estimator=self.keras_regressor,
                                            params=params,
                                            verbose=0,
                                            population_size=10,
                                            fit_params=fit_params)

        evo_hist = self.evo.fit(X=self.data_obj.x_train, y=self.data_obj.y_train)
        return evo_hist

