from wrapper import Wrap
from data_op import DataManager
from pandas import DataFrame

data_obj = DataManager('daily', 4, 6)

experiment_obj = Wrap(data_obj=data_obj, patience=8)

history_evo = experiment_obj.random_method()

print(history_evo.best_params_, history_evo.cv_results_)


# DataFrame(history_evo.hist.history).to_csv('history.csv')
