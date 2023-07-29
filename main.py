from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import  os
import numpy as np
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':

    for i in range(5):
        config_dict = {
            'gpu_id': 0,
            'epochs': 100,
            'n_layers': [4],
            'reg_weight': [0.01],
            'momentum': [0.05],
            'dropout': [0.1],
        }
        flod = i
        model = 'CLPiDA'
        dataset = 'dateset'
        save_model = True

        config = Config(model, dataset, config_dict)

        dataset = RecDataset(config, flod)
        train_dataset, test_dataset = dataset.split(config['split_ratio'])
        #str(train_dataset)
        train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=False)

        test_data = []
        ls = np.loadtxt('./train data/data_{}.inter'.format(flod),
                        dtype=int, skiprows=1)
        for i in range(len(ls)):
            if (ls[i][3] == 1):
                test_data.append([ls[i][0], ls[i][1]])

        hyper_ret = []
        val_metric = config['valid_metric'].lower()
        best_test_value = 0.0
        idx = best_test_idx = 0

        hyper_ls = []
        if "seed" not in config['hyper_parameters']:
            config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
        for i in config['hyper_parameters']:
            hyper_ls.append(config[i] or [None])
        combinators = list(product(*hyper_ls))
        total_loops = len(combinators)
        for hyper_tuple in combinators:
            for j, k in zip(config['hyper_parameters'], hyper_tuple):
                config[j] = k

            init_seed(config['seed'])

            train_data.pretrain_setup()

            model = get_model(config['model'])(config, train_data).to(config['device'])
            trainer = get_trainer()(config, model, flod)

            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=None, test_data=test_data,
                                                              saved=save_model)
        label, score, allscore, P_D = trainer.ROC(test_data)


