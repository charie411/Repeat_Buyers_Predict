import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklean.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn import svm, preprocessing
from sklearn.feature_selection import SelectKBest, chi2
import pickle, os
import matplotlib.pyplot as plt

#Define a function to train Random Forest Model
def RfModelDecision(train_x, train_y, test_x):
  rf_model = RandomForestClassifier(n_estimators=60)
  rf_model.fit(train_x,train_y)
  
  test_pred_proba_rf = rf_model.predict_proba(test_x)
  
  score = test_pred_proba_rf.tolist()
  score_rf = [x[1] for x in score]
  
  return score_rf, rf_model
  
#Define a function to train SVM Model
def SVMModelDecision(train_x, train_y, test_x):
  svm_model = svm.SVC(C=35, gamma=0.005, kernel='rbf', probability=True)
  svm_model.fit(train_x,train_y)
  
  test_pred_proba_svm = svm_model.predict_proba(test_x)
  
  score = test_pred_proba_svm.tolist() 
  score_svm = [x[1] for x in score]
  
  return score_svm, svm_model
  
#Define a function to print and plot roc and auc info 
def ROC_Plot(predict_vec, label):
  fpr, tpr, thresholds = roc_curve(label, predict_vec)
  roc_auc = auc(fpr, tpr)
  print('AUC:',roc_auc)
  plt.plot(fpr,tpr,lw=1)

#Define a function to Kfold validate models  
def stratifiedkf_model(feature, label, clf, models=[], scores=[]):
  folds = StratifiedKFold(n_splits=5, random_state=2019)
  
  for fold_, (trn_idx,val_idx) in enumerate(folds.split(feature,label)):
    print('fold n°{}'.format(fold_+1))
    
    train_x, val_x = feature[trn_idx], feature[cal_idx]
    train_y, val_y = label[trn_idx], label[cal_idx]
       
    tmp_score, model = clf(train_x, train_y, val_x)
    models.append(model)
    scores.append(tmp_score)
    
    #check training set accuracy
    train_pred = model.predict(train_x)
    tn, fp, fn, tp = confusion_matrix(y_pred=train_pred, y_true=train_y).ravel()
    print('training set accuracy ammounts: P:%s N:%s'%(tp, tn))
    
    #check validate set accuracy
    ROC_Plot(tmp_score, val_y)
  return models, scores, val_y
  
#Define a function to train, fit, submit results
def ModelMainFunc(operation):
  if operation == 'train':
    train_feat_org = pickle.load(open("./data/feature_train/feat_final.pytmp"))
    label = pickle.load(open("./data/feature_train/label.pytmp"))
#     scaler_file = "./data/model/scaler.pytmp"

#     if os.path.exists(scaler_file):
#       scaler = preprocessing.StandardScaler().fit(feature_all)
#       pickle.dump(scaler, open(scaler_file, 'wb'))
#     else:
#       scaler = pickle.load(open(scaler_file, 'wb'))

#     feat_scale = scaler.transform(feature_all)

    feat_scale_new = SelectKBest(chi2,k=23).fit(train_feat_org, label)
    train_feat_new = selectkbest.transform(train_feat_org)
    pickle.dump(selectkbest, open("./data/model/selectkbest.pytmp", 'wb'))

    #SVM model
    print('*' * 50)
    print('Start Building SVM model...')
    label = np.array(label)
    svm_models, svm_scores, test_label = stratifiedkf_model(train_feat_new, label, clf=SVMModelDecision, models=[], scores=[])

    svm_model_file = "./data/model/svm_model_files.pytmp"
    pickle.dump(obj=svm_models, file=open(svm_model_file, 'wb'))

    #Random Forest Model
    print('*' * 50)
    print('Start Building Random Forest model...')
    rf_models, rf_scores, test_label = stratifiedkf_model(train_feat_new, label, clf=RfModelDecision, models=[], scores=[])

    rf_model_file = "./data/model/rf_model_files.pytmp"
    pickle.dump(obj=rf_models, file=open(rf_model_file, 'wb')) 

    svm_scores_arr = np.array(svm_scores)
    rf_scores_arr = np.array(rf_scores)
    score_integ = (0.5 * (svm_scores_arr + rf_scores_arr)).tolist()
    for scr in score_integ:
      ROC_Plot(scr, test_label)

  if operation == 'test':
    test_feat_org = pickle.load(open("./data/feature/feat_final.pytmp",'rb'))

    scaler = pickle.load(open("./data/model/scaler.pytmp",'rb'))

    svm_models = pickle.load(open("./data/model/svm_model_files.pytmp",'rb'))
    rf_models = pickle.load((open("./data/model/rf_model_files.pytmp",'rb'))

    test_feat = scaler.transform(test_feat_org)

    print("svm predict...")
    score_total_svm = np.array([[0,0]] * len(test_feat))
    for svm_model in svm_models:
      score_tmp = svm_model.predict_proba(test_feat)
      score_total_svm += score_tmp / len(svm_models)

    print("random forest predict...")
    score_total_rf = np.array([[0,0]] * len(test_feat))
    for rf_model in rf_models:
      score_tmp = rf_model.predict_proba(test_feat)
      score_total_rf += score_tmp / len(rf_models)

    score_tmp = (score_total_svm + score_total_rf) / 2
    score_final = [np.round(x[1]) for x in score_tmp]

    #write result score into submit file 
    test_file = "./data/test_format1.csv"
    test_read = pd.read_csv(test_file)

    #Notice the order of predictions
    test_read['label'] = score_final
    result_file = './data/submit_result_allsample.csv'
    test_read.to_csv(result_file)
