import os
import numpy as np
import pickle
import json
import pandas as pd

class loader:
    '''
      Data Loader with initial pre-processing
      Arguments
      - dir_path (str)  : the path to the data directory (i.e. ./data/kindle)
      - json_file (str) : the name of the json file in the data directory(i.e. Kindle_Store_5.json)
      - seed (int)      : seed constant for splitting the data to train & test
     '''
    def __init__(self, dir_path, json_file, seed=2020):
        self.TPS_DIR = dir_path
        self.TP_file = os.path.join(dir_path, json_file)
        self.seed = seed
        
    def _get_count(self, tp, id):
        playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
        count = playcount_groupbyid.size()
        return count
    
    def _numerize(self, tp, user2id, item2id):
        uid = map(lambda x: user2id[x], tp['user_id'])
        sid = map(lambda x: item2id[x], tp['item_id'])
        tp['user_id'] = uid
        tp['item_id'] = sid
        return tp
        
    def load(self):
        print 'Data Loading Started.'
        f= open(self.TP_file)
        users_id=[]
        items_id=[]
        ratings=[]
        reviews=[]
        np.random.seed(self.seed)

        for line in f:
            js=json.loads(line)
            if str(js['reviewerID'])=='unknown':
                print "unknown"
                continue
            if str(js['asin'])=='unknown':
                print "unknown2"
                continue
            if 'reviewText' not in js:
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID'])+',')
            items_id.append(str(js['asin'])+',')
            ratings.append(str(js['overall']))
        data=pd.DataFrame({'user_id':pd.Series(users_id),
                           'item_id':pd.Series(items_id),
                           'ratings':pd.Series(ratings),
                           'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]

    
        usercount, itemcount = self._get_count(data, 'user_id'), self._get_count(data, 'item_id')
        unique_uid = usercount.index
        unique_sid = itemcount.index
        item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

        data=self._numerize(data, user2id, item2id)
        tp_rating=data[['user_id','item_id','ratings']]


        n_ratings = tp_rating.shape[0]
        test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
        test_idx = np.zeros(n_ratings, dtype=bool)
        test_idx[test] = True

        tp_1 = tp_rating[test_idx]
        tp_train= tp_rating[~test_idx]

        data2=data[test_idx]
        data=data[~test_idx]


        n_ratings = tp_1.shape[0]
        test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

        test_idx = np.zeros(n_ratings, dtype=bool)
        test_idx[test] = True

        tp_test = tp_1[test_idx]
        tp_valid = tp_1[~test_idx]
        tp_train.to_csv(os.path.join(self.TPS_DIR, 'train.csv'), index=False,header=None)
        tp_valid.to_csv(os.path.join(self.TPS_DIR, 'valid.csv'), index=False,header=None)
        tp_test.to_csv(os.path.join(self.TPS_DIR, 'test.csv'), index=False,header=None)

        user_reviews={}
        item_reviews={}
        user_rid={}
        item_rid={}
        for i in data.values:
            if user_reviews.has_key(i[0]):
                user_reviews[i[0]].append(i[3])
                user_rid[i[0]].append(i[1])
            else:
                user_rid[i[0]]=[i[1]]
                user_reviews[i[0]]=[i[3]]
            if item_reviews.has_key(i[1]):
                item_reviews[i[1]].append(i[3])
                item_rid[i[1]].append(i[0])
            else:
                item_reviews[i[1]] = [i[3]]
                item_rid[i[1]]=[i[0]]


        for i in data2.values:
            if user_reviews.has_key(i[0]):
                l=1
            else:
                user_rid[i[0]]=[0]
                user_reviews[i[0]]=['0']
            if item_reviews.has_key(i[1]):
                l=1
            else:
                item_reviews[i[1]] = [0]
                item_rid[i[1]]=['0']

        pickle.dump(user_reviews, open(os.path.join(self.TPS_DIR, 'user_review'), 'wb'))
        pickle.dump(item_reviews, open(os.path.join(self.TPS_DIR, 'item_review'), 'wb'))
        pickle.dump(user_rid, open(os.path.join(self.TPS_DIR, 'user_rid'), 'wb'))
        pickle.dump(item_rid, open(os.path.join(self.TPS_DIR, 'item_rid'), 'wb'))

        usercount, itemcount = self._get_count(data, 'user_id'), self._get_count(data, 'item_id')
        print 'Data Loader Finished...'
        print 'Files Saved at {}'.format(self.TPS_DIR)