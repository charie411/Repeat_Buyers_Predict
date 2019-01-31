import os
import pandas as pd
import numpy as np
import pickle

#define a function to split user_log_format1.csv into grouped subfiles 
def table_grouped_split(inPath,outPath,key):
  """
  INPUT:
      inPath -- path for user_log_format1.csv
      outPath -- path for grouped subfiles to store
      key -- grouping key
  OUTPUT:
  """
  chunksize = 5000000
  chunks = pd.read_csv(inPath,chunksize=chunksize)
  
  for chunk in chunks:
    chunk.columns = ['user_id', 'item_id', 'cat_id', 'seller_id', \
                       'brand_id', 'time_stamp','action_type']
    cg = chunk.groupby(key)
    for i in chunk[key].unique():
      df = cg.get_group(i)
      if os.path.exists(outPath.format(i)):
        df.to_csv(outPath.format(i),mode='a',index=False,header=False)
      else:
        df.to_csv(outPath.format(i),mode='a',index=False,header=True)
        
#==========FOR CATEGORY FILES==========
 #define a function to obtain category-feature statistics for each categorically grouped subfile
 def cate_statistic(filename,train,cate_feat):
  """
  INPUT:
      filename -- path for each categorically grouped subfile
      train -- pandas DataFrame for trainning data, which is used as filters of user_id & seller_id
               for category-feature statistics
      cate_feat -- category-feature dict which is recursively updated for each categorically grouped subfile
  OUTPUT:
      cate_feat -- category-feature dict which is recursively updated for each categorically grouped subfile
      data in cate_feat:
        [0] the share of each product buyed in the category
        [1] the total number of sellers in the category
        [2] the total number of repeated buyers in the category
        [3] the total sales in the category
  """
  cate_path = "./data/category/{}".format(filename)
  cate_logs = pd.read_csv(cate_path)
  
  #the total number of sellers in the category
  total_seller_cate = cate_logs.seller_id.nunique()
  
  #the total number of repeated buyers in the category
  cate_buy_logs = cate_logs.query('action_type == 2')
  
  if len(cate_buy_logs) < 1:
    return cate_feat
  num_repeat_buyer = cate_buy_logs.groupby('user_id').time_stamp.nunique().gt(1).sum()
  
  #the total sales in the category
  total_sale_cate = len(cate_buy_logs)
  
  #with train file to obtain the share of each product buyed in the category
  samp_keys = train.apply(lambda row:'%d+%d'%(row.iloc[0],row.iloc[1]),axis=1)
  cate_buy_logs['key_usr_slr'] = cate_buy_logs.apply(lambda row:'%d+%d'%(row.iloc[0],row.iloc[3]),axis=1)
  cate_buy_logs['key_itm_slr_cat_usr'] = cat_buy_logs.apply(\
  lambda row:'%d+%d+%d+%d'%(row.iloc[1],row.iloc[3],row.iloc[2],row.iloc[0]),axis=1)
  
  cate_train_logs = cate_buy_logs[cate_buy_logs.key_usr_slr.isin(samp_keys.values)]
  cate_dict = cate_train_logs.groupby(['seller_id','item_id']).size().to_dict()
  cate_train_logs_key_set = set(cate_train_logs['key_itm_slr_cat_usr'])
  
  for k in cate_train_logs_key_set:
    cate_feat[k] = [cate_dict[(int(k.split('+')[1]),int(k.split('+')[0]))]/total_sale_cate,\
    total_seller_cate, num_repeat_buyer, total_sale_cate]
  print(filename,'is over.')
  return cate_feat  
#==============FOR SELLER FILES======
#the function deals with seller_grouped or seller_item_groued data,output seller features
def _ClickandCollect(seller_logs,usr_slr_iter=False):
    """
    INPUT:
        seller_logs --  seller log pandas DataFrame
        usr_slr_iter -- if usr_slr_iter is True,then the function deals with usr_slr_groued
                        data,results returned differs. default:False
    OUTPUT:
        usr_slr_iter == True:
            days_click_usr -- the number of days of clicking for each item
            num_collect_seller -- the number of collection in the seller for each item
            num_cart_seller -- the number of being added to cart in the seller for each item
            total_sales -- the sales in the seller for each item
        usr_slr_iter == False:
            total_sales -- the total sales in the seller
            num_usr_buy -- the total number of buyer in the seller
            num_repeated_buyer -- the total number of repeated buyer in the seller
            num_collect_seller -- the total number of collection in the seller
            num_cart_seller -- the total number of being added to cart in the seller
    """
    #the total number of collection in the seller
    num_collect_seller = len(seller_logs.query('action_type == 3'))
    
    #the total number of being added to cart in the seller
    num_cart_seller = len(seller_logs.query('action_type == 1'))
    
    #the total sales in the seller
    usr_log_buy = seller_logs.query('action_type == 2')
    total_sales = len(usr_log_buy)
    
    if usr_slr_iter:
        #the total number of days of clicking for each item,the total number of collecting for eath item
        #the total number of adding to cart for each item,the total number of purchase for each item    
        days_click_usr = usr_log_buy.time_stamp.nunique()
        return days_click_usr, num_collect_seller, num_cart_seller,total_sales
    else:
        #the total number of buyer in the seller
        num_usr_buy = usr_log_buy['user_id'].nunique()
        
        #the total number of repeated buyer in the seller
        grouped = usr_log_buy.groupby('user_id')
        num_repeated_buyer = grouped.time_stamp.nunique().gt(1).sum()
        return total_sales, num_usr_buy, num_repeated_buyer, num_collect_seller, num_cart_seller     
#======
#define a function to obtain seller-feature statistics for each seller grouped subfile
def seller_statistic(filename,train,usr_info_read,seller_dict,cate_feat):
    """
    INPUT:
        filename -- path for each seller grouped subfile
        train -- pandas DataFrame for trainning data, which is used as filters of user_id & seller_id
                 for category-feature statistics
        usr_info_read -- pandas DataFrame for user-info data, which is used to provide age_range & gender
                 for following user-feature statistics
        seller_dict -- seller-feature dict which is recursively updated for each seller grouped subfile
        cate_feat -- category-feature dict
    OUTPUT:
        seller_dict -- seller-feature dict which is recursively updated for each seller grouped subfile
        usr_gender_age_range -- pandas DataFrame of buy logs for training users  
                                and columns contains user_id, age_range, gender
        data in seller_dict:
            [0] total_sales: the total sales for the seller
            [1] num_usr_buy: the total number of distinct buyers in the seller
            [2] num_repeated_buyer: the total number of repeated buyers in the seller
            [3] num_collect_seller: the total number of collecting in the seller
            [4] num_cart_seller: the total number of adding into cart in the seller
            [5] seller_mode_age_range: the majority age_range in the seller
            [6] seller_mode_gender: the majority gender in the seller
            [7] days_click_usr: the date number of clicking for the specific buyer in the seller
            [8] num_collect_usr: the number of collecting for the specific buyer in the seller
            [9] num_cart_usr: the number of adding into cart for the specific buyer in the seller
            [10] num_buy_usr: the number of buying for the specific buyer in the seller
            [11] max_sales_product: the sales for each item in the seller and chooses the max sales
            [12] min_sales_product: the sales for each item in the seller and chooses the min sales
            [13] mean_sales_product: the sales for each item in the seller and chooses the mean sales
            [14] max_num_repeated_product: 
            [15] min_num_repeated_product: 
            [16] mean_num_repeated_product: 
            [17] max_cate_feat_product_usr_0: the share of each product buyed in the slr_usr and choose the max share
            [18] min_cate_feat_product_usr_0: the share of each product buyed in the slr_usr and choose the min share
            [19] mean_cate_feat_product_usr_0: the share of each product buyed in the slr_usr and choose the mean share
            [20] cate_feat_product_usr_1: the total number of sellers in the category?
            [21] cate_feat_product_usr_2: the total number of repeated buyers in the category?
            [22] cate_feat_product_usr_3: the total sales in the category?
    """
    seller_path = "./data/seller_id/{}".format(filename)
    seller_logs = pd.read_csv(seller_path)
    
    total_sales, num_usr_buy, num_repeated_buyer, num_collect_seller, num_cart_seller \
    = _ClickandCollect(seller_logs,usr_slr_iter=False)
    
    samp_keys = train.apply(lambda row:'%d+%d'%(row.iloc[0],row.iloc[1]),axis=1)
    seller_buy_logs = seller_logs.query('action_type == 2')
    seller_buy_logs['key_usr_slr'] = seller_buy_logs.apply(lambda row:'%d+%d'%(row.iloc[0],row.iloc[3]),axis=1)
    seller_buy_logs['key_itm_slr_cat_usr'] = seller_buy_logs.apply(\
    lambda row:'%d+%d+%d+%d'%(row.iloc[1],row.iloc[3],row.iloc[2],row.iloc[0]),axis=1)
  
    seller_buy_logs = seller_buy_logs.join(usr_info_read.set_index('user_id'))
    seller_mode_age_range = seller_buy_logs.age_range.mode()[0]
    seller_mode_gender = seller_buy_logs.gender.mode()[0]
    seller_train_logs = seller_buy_logs[seller_buy_logs.key_usr_slr.isin(samp_keys.values)]
    seller_train_logs.fillna(value={'age_range':seller_mode_age_range,'gender':seller_mode_gender},inplace=True)
    seller_train_logs_key_set = set(seller_train_logs['key_usr_slr'])
    
    #the sales(min,max,mean) of the product buyed, if product number is greater than 2 
    #then gets the aggregation max-min-mean values
    max_sales_product, min_sales_product, mean_sales_product = seller_train_logs.groupby('item_id').size()\
    .agg(['max','min','mean'])
    
    #the number(min,max,mean) of repeated buy
    asd = seller_train_logs.groupby(['item_id','user_id']).time_stamp.nunique().rename('num_date').reset_index()
    max_num_repeat_product, min_num_repeat_prodct, mean_num_repeat_product = asd.groupby('item_id').num_date.\
    agg(lambda x:sum(x>1)).agg(['max','min','mean'])
    
    for k in seller_train_logs_key_set:
        d = seller_train_logs.query('key_usr_slr==@k')
        days_click_usr, num_collect_usr, num_cart_usr, num_buy_usr = _ClickandCollect(d,usr_slr_iter=True)
        cate_feat_product_usr = [cate_feat[i] for i in set(d.key_itm_slr_cat_usr)]
        
        cate_feat_product_usr_0 = [i[0] for i in cate_feat_product_usr]
        max_cate_feat_product_usr_0, min_cate_feat_product_usr_0, mean_cate_feat_product_usr_0 = \
        max(cate_feat_product_usr_0), min(cate_feat_product_usr_0), np.mean(cate_feat_product_usr_0)
        cate_feat_product_usr_1 = max([i[1] for i in cate_feat_product_usr])
        cate_feat_product_usr_2 = max([i[2] for i in cate_feat_product_usr])
        cate_feat_product_usr_3 = max([i[3] for i in cate_feat_product_usr])
        
        seller_dict[k] = [total_sales,num_usr_buy,num_repeated_buyer,num_collect_seller,num_cart_seller,\
        seller_mode_age_range,seller_mode_gender,\
        days_click_usr, num_collect_usr,num_cart_usr,num_buy_usr,\
        max_sales_product,min_sales_product,mean_sales_product,\
        max_num_repeated_product,min_num_repeated_product,mean_num_repeated_product,\
        max_cate_feat_product_usr_0, min_cate_feat_product_usr_0, mean_cate_feat_product_usr_0,
        cate_feat_product_usr_1,cate_feat_product_usr_2,cate_feat_product_usr_3]
    print(filename,'is over.')
    usr_gender_age_range = seller_train_logs[usr_info_read.columns]
    return seller_dict, usr_gender_age_range
#==============FOR USER FILES======
def usr_statistic(filename, usr_gender_age_range_all, usr_dict, usr_info_read):
    """
    INPUT:
        filename -- path for each user grouped subfile
        usr_gender_age_range_all -- pandas DataFrame, which is used to provide age_range & gender
                                    for user-feature statistics with na-values filled
        usr_dict -- user-feature dict which is recursively updated for each user grouped subfile
        usr_info_read -- pandas DataFrame for user-info data
    OUTPUT：
        usr_dict -- user-feature dict which is recursively updated for each user grouped subfile
        data in usr_dict:
            [0] usr_buy_days: the date number of buying for the user
            [1] num_repeatedbuy: the number of repeated buying for the user
            [2] num_buys: the number of buying for the user
            [3] num_collect: the number of collecting for the user
            [4] num_cart: the number of adding into cart for the user
            [5] usr_click_days: the date number of clicking for the user
            [6] age_range: the age range of the user
            [7] gender: the gender of the user
    """
    usr_path = "./data/usr_id/{}".format(filename)
    usr_logs = pd.read_csv(seller_path)
    usr_id = usr_logs.user_id.unique()[0]
    
    usr_log_buy = usr_logs.query('action_type == 2')
    usr_log_click = usr_logs.query('action_type == 0')
    
    usr_buy_days = usr_log_buy.time_stamp.nunique()
    num_repeatedbuy = usr_log_buy.groupby('seller_id').time_stamp.nunique().gt(1).sum()
    
    num_buys = len(usr_log_buy)
    num_collect = len(usr_logs.query('action_type == 3'))
    num_cart = len(usr_logs.query('action_type == 1'))
    num_click = len(usr_log_click)
    usr_click_days = usr_log_click.time_stamp.nunique()
    
    sll = usr_gender_age_range_all.query('user_id == @usr_id')
    asd = usr_info_read.query('user_id == @usr_id')
    
    try:
        age_range = sll.age_range.mode()[0]
    except:
        age_range = asd.age_range.mode()[0]
    
    try:
        gender = sll.gender.mode()[0]
    except:
        gender = asd.gender.mode()[0] 
        
    usr_dict[str(usr_id)] = [usr_buy_days, num_repeatedbuy, num_buys, num_collect, num_cart, usr_click_days,age_range,gender]
    print(filename,'is over.')
    return usr_dict    
#==============FOR FINAL FEATURES======
def genFinalFeat(train):
    print("Generate the final features...")
    
    #read seller feature from dict_seller file
    seller_file = './data/feature/feat_seller.pytmp'
    seller_feat = pickle.load(open(seller_file,'r'))
    
    #read user feature from dict_user file
    usr_file = './data/feature/feat_usr.pytmp'
    usr_feat = pickle.load(open(usr_file,'r'))  
    
    #combine seller feature with user feature
    train['key_usr_slr'] = train.apply(lambda row:'%d+%d'%(row.iloc[0],row.iloc[1]),axis=1)
    train['finalfeat'] = train.apply(lambda x:usr_feat[str(x['user_id'])]+seller_feat[x['key_usr_slr']],axis=1)
    finalfeat = train['finalfeat'].tolist()
    return finalfeat    
#============
usrlog_file = "./data/user_log_format1.csv"
files_output = ["./data/usr_id/{}.csv","./data/category/{}.csv","./data/seller_id/{}.csv"]
keys = ['user_id','cat_id','seller_id']

for key,outPath in zip(keys,files_output):
  table_grouped_split(inPath=usrlog_file, outPath=outPath, key=key)
#====================
print("Category Feature Extract...")
train_file = './data/train_format1.csv'
train_read = pd.read_csv(train_file)

cate_feat = {}
for filename in os.listdir("./data/category"):
    cate_feat = cate_statistic(filename=filename, train=train_read, cate_feat=cate_feat)

cate_file = "./data/feature/feat_cate.pytmp"
pickle.dump(obj=cate_feat, file=open(cate_file,"wb"))
#====================
print("Seller Feature Extract...")
usr_info_file = './data/user_info_format1.csv'
usr_info_read = pd.read_csv(usr_info_file)

#read cate feature from dict_cate file
cate_file = "./data/feature/feat_cate.pytmp"
cate_feat = pickle.load(open(cate_file, 'rb'))

seller_dict = {}
ss = []
for filename in os.listdir("./data/seller_id"):
    seller_dict,seller_train_logs = seller_statistic(filename=filename, train=train_read, usr_info_read=usr_info_read,\
    seller_dict=seller_dict, cate_feat=cate_feat)
    ss.append(seller_train_logs)
 usr_gender_age_range_all = pd.cancat(ss)#data with gender/age_range na-filled 
 
seller_file = "./data/feature/feat_seller.pytmp"
pickle.dump(obj=seller_dict, file=open(seller_file,"wb"))
usr_gender_age_range_all_file = "./data/feature/usr_gender_age_range_all.pytmp"
pickle.dump(obj=usr_gender_age_range_all, file=open(usr_gender_age_range_all_file,"wb"))
 #====================
print("User Feature Extract...")

usr_dict = {}
for filename in os.listdir("./data/usr_id"):
    usr_dict = usr_statistic(filename=filename, usr_gender_age_range_all=usr_gender_age_range_all, usr_dict=usr_dict,\
    usr_info_read=usr_info_read)

usr_file = "./data/feature/feat_usr.pytmp"
pickle.dump(obj=usr_dict, file=open(usr_file,"wb"))
#=================RUNNING FUNCTIONS========
if __name__ == "__main__":
    feature_all = genFinalFeat(train_read)   
    save_feat = './data/feature/feat_final.pytmp'
    pickle.dump(obj=feature_all, file=open(save_feat,"wb"))
