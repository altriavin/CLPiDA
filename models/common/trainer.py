import os
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time


class AbstractTrainer(object):

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):

    def __init__(self, config, model,flod):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        print(self.epochs)
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.k = flod
        # save model
        self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config['eval_type']

        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000], gamma=0.001)
        depent_test = np.vstack((np.loadtxt('./Independent test data/positive.csv', dtype=int),
                                 np.loadtxt('./Independent test data/negative.csv', dtype=int)))
        test_pd = np.loadtxt('./Independent test data/PD.csv', dtype=int, delimiter=',')
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                print(epoch_idx+1)
                self.ROC(test_data)

                #self.calculate_dependttest(depent_test, test_pd)
                scheduler.step()
        return self.best_valid_score, self.best_valid_result


    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, is_test=False, idx=0):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_diseases = batched_data[1]
            scores[masked_diseases[0], masked_diseases[1]] = -(1 << 10)
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def ROC(self,eval_data):
        batchedy_data=[]
        list=[]
        for i in range(10149):
            list.append(i)
        batchedy_data.append(list)
        scores = self.model.full_sort_predict(batchedy_data)
        label=[]
        input=[]
        score=scores.detach().cpu().numpy()
        P_D = []
        for i in range(len(eval_data)):
            label.append(1)
            input.append(score[eval_data[i][0]][eval_data[i][1]])
            P_D.append([eval_data[i][0],eval_data[i][1],score[eval_data[i][0]][eval_data[i][1]]])
        negtest=np.loadtxt('C:\作业\对比实验\CLPiDA/train data/negitive_test_{}.csv'.format(self.k),dtype=int)
        for i in range(len(negtest)):
            label.append(0)
            input.append(score[negtest[i][0]][negtest[i][1]])
            P_D.append([negtest[i][0], negtest[i][1], score[negtest[i][0]][negtest[i][1]]])
        tst_auc, tst_aupr=self.tst_metric(label,input)
        print("auc:{},   aupr:{}".format(tst_auc,tst_aupr))
        return label,input,score,P_D

    def getemb(self):
        u_online, u_target, i_online, i_target=self.model.getembding()
        return u_online, u_target, i_online, i_target
    def tst_metric(self,label, input):
        return roc_auc_score(label, input), average_precision_score(label, input)

    def calculate_dependttest(self,depent_test,test_pd):
        scores = self.model.depenttest(test_pd)
        scores = scores.detach().cpu().numpy()
        input = []
        label = []
        for i in range(len(depent_test)):
            x = depent_test[i][0]
            y = depent_test[i][1]
            if(test_pd[x][y] == 1):
                label.append(1)
            else:
                label.append(0)
            input.append(scores[x][y])
        tst_auc, tst_aupr = self.tst_metric(label, input)
        np.save('./dependent_test/label.npy',np.array(label))
        np.save('./dependent_test/score.npy',np.array(input))
        print("depend test:auc:{},   aupr:{}".format(tst_auc, tst_aupr))