import math
import torch
import random
import numpy as np
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config['device']
        print(self.dataset.inter_num)
        print(self.dataset.piRNA_num)
        print(self.dataset.disease_num)
        self.sparsity = 1 - self.dataset.inter_num / self.dataset.piRNA_num / self.dataset.disease_num
        self.pr = 0
        self.inter_pr = 0

    def pretrain_setup(self):

        pass

    def data_preprocess(self):
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


class TrainDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=None,batch_size=batch_size, neg_sampling=True, shuffle=shuffle)

        self.history_diseases_per_u = dict()
        self.all_diseases = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_disease_len = len(self.all_diseases)
        self.use_full_sampling = config['use_full_sampling']

        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_uids_sample
            else:
                self.sample_func = self._get_neg_sample
        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_diseases_u()

    def pretrain_setup(self):
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        self.all_diseases.sort()
        if self.use_full_sampling:
            self.all_uids.sort()
        random.shuffle(self.all_diseases)

    def inter_matrix(self, form='coo', value_field=None):
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(10149, 19))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        if self.use_full_sampling:
            return len(self.all_uids)
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        piRNA_tensor = torch.tensor(cur_data[self.config['PIRNA_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        disease_tensor = torch.tensor(cur_data[self.config['DISEASE_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(piRNA_tensor, 0),
                                  torch.unsqueeze(disease_tensor, 0)))
        u_ids = cur_data[self.config['PIRNA_ID_FIELD']]
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)
        batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))
        return batch_tensor

    def _get_non_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        # to tensor
        piRNA_tensor = torch.tensor(cur_data[self.config['PIRNA_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        disease_tensor = torch.tensor(cur_data[self.config['DISEASE_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(piRNA_tensor, 0),
                                  torch.unsqueeze(disease_tensor, 0)))
        return batch_tensor

    def _get_full_uids_sample(self):
        piRNA_tensor = torch.tensor(self.all_uids[self.pr: self.pr + self.step]).type(torch.LongTensor).to(self.device)
        self.pr += self.step
        return piRNA_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        for u in u_ids:
            # random 1 disease
            iid = self._random()
            while iid in self.history_diseases_per_u[u]:
                iid = self._random()
            neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def _random(self):
        rd_id = random.sample(self.all_diseases, 1)[0]
        return rd_id

    def _get_history_diseases_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u, u_ls in uid_freq:
            self.history_diseases_per_u[u] = u_ls.values
        return self.history_diseases_per_u


class EvalDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_diseases_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        self.pos_diseases_per_u = self._get_pos_diseases_per_u(self.eval_u).to(self.device)
        self._get_eval_diseases_per_u(self.eval_u)
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr: self.pr+self.step])
        batch_piRNAs = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_matrix = self.pos_diseases_per_u[:, self.inter_pr: self.inter_pr+inter_cnt].clone()
        batch_mask_matrix[0] -= self.pr
        self.inter_pr += inter_cnt
        self.pr += self.step

        return [batch_piRNAs, batch_mask_matrix]

    def _get_pos_diseases_per_u(self, eval_piRNAs):
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        uid_freq = self.additional_dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_piRNAs):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i]*i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def _get_eval_diseases_per_u(self, eval_piRNAs):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u in eval_piRNAs:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_diseases_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    def get_eval_diseases(self):
        return self.eval_diseases_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_piRNAs(self):
        return self.eval_u.cpu()


