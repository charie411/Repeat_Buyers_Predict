import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklean.metrics import roc_auc_score, roc_curve, auc
from sklearn import svm, preprocessing
from sklearn.feature_selection import SelectKBest, chi2
import pickle, os
import matplotlib.pyplot as plt

#Define a function to distinguish positive from negative instances
def sortClass(feature, label):
  samp_feat_p, samp_feat_n = [], []
  
  for smp,lbl in zip(feature,label):
  if lbl == 1:
    samp_feat_p.append(smp)
  elif lbl == 0:
    dsmp_feat_n.append(smp)
  return samp_feat_p, samp_feat_n
  
#Define a function to train Random Forest Model
def RfModelDecision(train_x, label, Ptest_x, Ntest_x):
  rf_model = RandomForestClassifier(n_estimators=60)
  rf_model.fit(train_x,label)
  
  Ppred_rf = rf_model.predict_proba(Ptest_x)
  Npred_rf = rf_model.predict_proba(Ntest_x)
  
  test_label = [1] * len(Ppred_rf) + [0] * len(Npred_rf)
  score = Ppred_rf.tolist() +Npred_rf.tolist()
  score_rf = [x[rf_model.classes_.argmax()] for x in score]
  ROC_Plot(score_rf, test_label)
  auc = roc_auc_score(test_label, score_rf)
  print('rf_AUC:', auc)
  
  return score_rf, rf_model, test_label
  
#Define a function to train SVM Model
def SVMModelDecision(train_x, label, Ptest_x, Ntest_x):
  svm_model = svm.SVC(C=35, gamma=0.005, kernel='rbf', probability=True)
  svm_model.fit(train_x,label)
  
  Ppred_svm = svm_model.predict_proba(Ptest_x)
  Npred_svm = svm_model.predict_proba(Ntest_x)
  
  test_label = [1] * len(Ppred_svm) + [0] * len(Npred_svm)
  score = Ppred_svm,f.tolist() +Npred_svm.tolist()
  score_svm = [x[svm_model.classes_.argmax()] for x in score]
  ROC_Plot(score_svm, test_label)
  auc = roc_auc_score(test_label, score_svm)
  print('rf_AUC:', auc)
  
  return score_svm, svm_model, test_label
  
#Define a function to print and plot roc and auc info 
def ROC_Plot(predict_vec, label):
  fpr, tpr, thresholds = roc_curve(label, predict_vec)
  roc_auc = suc(fpr, tpr)
  print('AUC:',auc)
  plt.plot(fpr,tpr,lw=1)

#Define a function to Kfold validate models  
def kf_model(X_p, X_n, clf, models=[], scores=[]):
  folds =KFold(n_splits=5, random_state=2019)
  
  for fold_, (trn_idx,val_idx) in enumerate(folds.split(X_p,X_n)):
    print('fold n°{}'.format(fold_+1))
    
    X_p_train, X_p_val = X_p[trn_idx], X_p[cal_idx]
    X_n_train, X_n_val = X_n[trn_idx], X_n[cal_idx]
    train_x = X_p_train + X_n_train
    train_y = [1] * len(X_p_train) + [0] * len(y_n_train)
    
    tmp_score, model, test_label = clf(train_x, train_y, X_p_val, X_n_val)
    models.append(model)
    scores.append(tmp_score)
    
    #check training set accuracy
    sum_Ptrain = sum(model.predict(X_p_train))
    sum_Ntrain = len(X_n_train) -sum(model.predict(X_n_train))
    print('training set accuracy ammounts: P:%s N:%s'%(sum_Ptrain, sum_Ntrain))
  return models, scores, test_label
  
#Define a function to train, fit, submit results
def ModelMainFunc(operation):
  if operation == 'train':
    feature_all = pickle.load(open("./data/feature_train/feat_final.pytmp"))
    label = pickle.load(open("./data/feature_train/label.pytmp"))
    scaler_file = "./data/model/scaler.pytmp"

    if os.path.exists(scaler_file):
      scaler = preprocessing.StandardScaler().fit(feature_all)
      pickle.dump(scaler, open(scaler_file, 'wb'))
    else:
      scaler = pickle.load(open(scaler_file, 'wb'))

    feat_scale = scaler.transform(feature_all)

    feat_scale_new = SelectKBest(chi2,k=23).fit_transform(feat_scale, label)
    X_p, X_n = sortClass(feat_scale_new, label)

    #SVM model
    print('*' * 50)
    print('Start Building SVM model...')
    svm_models, svm_scores, test_label = kf_model(X_p, X_n, clf=SVMModelDecision, models=[], scores=[])

    svm_model_file = "./data/model/svm_model_files.pytmp"
    pickle.dump(obj=svm_models, file=open(svm_model_file, 'wb'))

    #Random Forest Model
    print('*' * 50)
    print('Start Building Random Forest model...')
    rf_models, rf_scores, test_label = kf_model(X_p, X_n, clf=RfModelDecision, models=[], scores=[])

    rf_model_file = "./data/model/rf_model_files.pytmp"
    pickle.dump(obj=rf_models, file=open(rf_model_file, 'wb')) 

    svm_scores_arr = np.array(svm_scores)
    rf_scores_arr = np.array(rf_scores)
    score_integ = (0.5 * (svm_scores_arr + rf_scores_arr)).tolist()
    auc = roc_auc_score(test_label, score_integ)
    for scr in score_integ:
      ROC_Plot(scr, test_label)
    print('total_AUC:', auc)

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
