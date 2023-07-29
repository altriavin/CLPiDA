from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np

class RecDataset(object):
    def __init__(self, config,flod,df=None):
        self.config = config
        self.dataset_path = os.path.abspath(config['data_path'])
        self.preprocessed_dataset_path = os.path.abspath(config['preprocessed_data'])
        self.preprocessed_loaded = False
        self.logger = getLogger()
        self.dataset_name = config['dataset']
        self.k = flod

        self.uid_field = self.config['PIRNA_ID_FIELD']
        self.iid_field = self.config['DISEASE_ID_FIELD']
        self.ts_id = self.config['TIME_FIELD']

        if df is not None:
            self.df = df
            return
        self.ui_core_splitting_str = self._k_core_and_splitting()
        self.processed_data_name = 'data_{}.inter'.format(self.k)

        if self.config['load_preprocessed'] and self._load_preprocessed_dataset():
            self.preprocessed_loaded = True
            self.logger.info('\nData loaded from preprocessed dir: ' + self.preprocessed_dataset_path + '\n')
            return

        self._from_scratch()

        self._data_processing()

    def _k_core_and_splitting(self):
        piRNA_min_n = 1
        disease_min_n = 1
        if self.config['min_piRNA_inter_num'] is not None:
            piRNA_min_n = max(self.config['min_piRNA_inter_num'], 1)
        if self.config['min_disease_inter_num'] is not None:
            disease_min_n = max(self.config['min_disease_inter_num'], 1)

        ratios = self.config['split_ratio']
        tot_ratio = sum(ratios)

        ratios = [i for i in ratios if i > .0]
        ratios = [str(int(_ * 10 / tot_ratio)) for _ in ratios]
        s = ''.join(ratios)
        return 'u{}i{}_s'.format(piRNA_min_n, disease_min_n) + s

    def _load_preprocessed_dataset(self):
        file_path = os.path.join(self.preprocessed_dataset_path, self.processed_data_name)
        if not os.path.isfile(file_path):
            return False
        self.df = self._load_df_from_file(file_path, self.config['load_cols']+[self.config['preprocessed_data_splitting']])
        return True

    def _from_scratch(self):
        self.logger.info('Loading {} from scratch'.format(self.__class__))
        file_path = os.path.join(self.dataset_path, '{}.inter'.format(self.dataset_name))
        if not os.path.isfile(file_path):
            raise ValueError(  'File {} not exist'.format(file_path))
        self.df = self._load_df_from_file(file_path, self.config['load_cols'])

    def _load_df_from_file(self, file_path, load_columns):
        cnt = 0
        with open(file_path, 'r') as f:
            head = f.readline()[:-1]
            field_separator = self.config['field_separator']
            for field_type in head.split(field_separator):
                if field_type in load_columns:
                    cnt += 1
            if cnt != len(load_columns):
                raise ValueError('File {} lost some required columns.'.format(file_path))

        df = pd.read_csv(file_path, sep=self.config['field_separator'], usecols=load_columns)
        return df

    def _data_processing(self):
        self.df.dropna(inplace=True)

        self.df.drop_duplicates(inplace=True)

        self._filter_by_k_core(self.df)

        self._reset_index(self.df)

    def _filter_by_k_core(self, df):
        while True:
            ban_piRNAs = self._get_illegal_ids_by_inter_num(df, field=self.uid_field,
                                                           max_num=self.config['max_piRNA_inter_num'],
                                                           min_num=self.config['min_piRNA_inter_num'])
            ban_diseases = self._get_illegal_ids_by_inter_num(df, field=self.iid_field,
                                                           max_num=self.config['max_disease_inter_num'],
                                                           min_num=self.config['min_disease_inter_num'])

            if len(ban_piRNAs) == 0 and len(ban_diseases) == 0:
                return

            dropped_inter = pd.Series(False, index=df.index)
            if self.uid_field:
                dropped_inter |= df[self.uid_field].isin(ban_piRNAs)
            if self.iid_field:
                dropped_inter |= df[self.iid_field].isin(ban_diseases)

            df.drop(df.index[dropped_inter], inplace=True)

    def _get_illegal_ids_by_inter_num(self, df, field, max_num=None, min_num=None):

        if field is None:
            return set()
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or -1

        ids = df[field].values
        inter_num = Counter(ids)
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}

        self.logger.debug('[{}] illegal_ids_by_inter_num, field=[{}]'.format(len(ids), field))
        return ids

    def _reset_index(self, df):
        if df.empty:
            raise ValueError('Some feat is empty, please check the filtering settings.')
        df.reset_index(drop=True, inplace=True)

    def split(self, ratios):
        if self.preprocessed_loaded:
            dfs = []
            splitting_label = self.config['preprocessed_data_splitting']
            # splitting into training/validation/test
            for i in range(2):
                temp_df = self.df[self.df[splitting_label] == i].copy()
                temp_df.drop(splitting_label, inplace=True, axis=1)
                dfs.append(temp_df)
            # wrap as RecDataset
            full_ds = [self.copy(_) for _ in dfs]
            return full_ds

        tot_ratio = sum(ratios)
        # remove 0.0 in ratios
        ratios = [i for i in ratios if i > .0]
        ratios = [_ / tot_ratio for _ in ratios]


        split_ratios = np.cumsum(ratios)[:-1]
        split_timestamps = list(np.quantile(self.df[self.ts_id], split_ratios))

        df_train = self.df.loc[self.df[self.ts_id] < split_timestamps[0]]

        uni_piRNAs = pd.unique(df_train[self.uid_field])
        uni_diseases = pd.unique(df_train[self.iid_field])

        u_id_map = {k: i for i, k in enumerate(uni_piRNAs)}
        i_id_map = {k: i for i, k in enumerate(uni_diseases)}
        self.df[self.uid_field] = self.df[self.uid_field].map(u_id_map)
        self.df[self.iid_field] = self.df[self.iid_field].map(i_id_map)

        self.df.dropna(inplace=True)

        self.df = self.df.astype(int)


        dfs = []
        start = 0
        for i in split_timestamps:
            dfs.append(self.df.loc[(start <= self.df[self.ts_id]) & (self.df[self.ts_id] < i)].copy())
            start = i

        dfs.append(self.df.loc[start <= self.df[self.ts_id]].copy())


        self._save_dfs_to_disk(u_id_map, i_id_map, dfs)

        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def _save_dfs_to_disk(self, u_map, i_map, dfs):
        if self.config['load_preprocessed'] and not self.preprocessed_loaded:
            dir_name = self.preprocessed_dataset_path
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            u_df = pd.DataFrame(list(u_map.diseases()), columns=[self.uid_field, 'new_id'])
            i_df = pd.DataFrame(list(i_map.diseases()), columns=[self.iid_field, 'new_id'])
            u_df.to_csv(os.path.join(self.preprocessed_dataset_path,
                                     '{}_u_{}_mapping.csv'.format(self.dataset_name, self.ui_core_splitting_str)),
                        sep=self.config['field_separator'], index=False)
            i_df.to_csv(os.path.join(self.preprocessed_dataset_path,
                                     '{}_i_{}_mapping.csv'.format(self.dataset_name, self.ui_core_splitting_str)),
                        sep=self.config['field_separator'], index=False)

            for i, temp_df in enumerate(dfs):
                temp_df[self.config['preprocessed_data_splitting']] = i
            temp_df = pd.concat(dfs)
            temp_df.to_csv(os.path.join(self.preprocessed_dataset_path, self.processed_data_name),
                           sep=self.config['field_separator'], index=False)

    def copy(self, new_df):
        nxt = RecDataset(self.config,self.k,new_df)
        return nxt

    def num(self, field):
        if field not in self.config['load_cols']:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        uni_len = len(pd.unique(self.df[field]))
        return uni_len

    def shuffle(self):
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def sort_by_chronological(self):
        self.df.sort_values(by=[self.ts_id], inplace=True, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        if self.uid_field:
            self.piRNA_num = len(uni_u)
            self.avg_actions_of_users = self.inter_num/self.piRNA_num
            info.extend(['The number of piRNAs: {}'.format(self.piRNA_num),
                         'Average actions of piRNAs: {}'.format(self.avg_actions_of_users)])
        if self.iid_field:
            self.disease_num = len(uni_i)
            self.avg_actions_of_items = self.inter_num/self.disease_num
            info.extend(['The number of piRNAs: {}'.format(self.piRNA_num),
                         'Average actions of diseases: {}'.format(self.avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / self.piRNA_num / self.disease_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
