"""
to accomodate several types of model evaluation by printing out statistics, plotting roc curve
"""



from sklearn.metrics import classification_report, roc_auc_score, roc_curve, recall_score, accuracy_score, \
    confusion_matrix, average_precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


def evaluate(y_true, X, model, threshold, func='predict_proba'):
    """
    Evalute model performance
    :param y_true: array like, true label
    :param X: dataset to be evaluated
    :param model: fitted model
    :param threshold: float, threshold for determining predicted label
    :param func: decision function, e.g. predict_proba or decision_function or predict
    :return:
    """
    if func == 'predict_proba':  # e.g. logistic regression
        y_probas = model.predict_proba(X)[:, 1]
    elif func == 'decision_function':  # e.g. SVM
        y_probas = model.decision_function(X)
    elif func == 'predict':  # for NN
        y_probas = [r[0] for r in model.predict(X)]
    else:
        raise
    y_preds = np.where(np.array(y_probas) >= threshold, 1, 0)
    # print(y_preds)
    accuracy = accuracy_score(y_true, y_preds)
    auc = roc_auc_score(y_true, y_probas)
    recall = recall_score(y_true, y_preds)
    average_precision = average_precision_score(y_true, y_preds, average='micro')
    f1 = f1_score(y_true, y_preds)
    
    result_dict = {'Accuracy':accuracy,'AUC':auc,'Recall':recall,'F1':f1,'AP':average_precision,'Confusion_Matrix':confusion_matrix(y_true,y_preds)}
    
    fpr, tpr, _ = roc_curve(y_true, y_probas)
    plt.plot(fpr, tpr, label='auc=' + str(auc), color='orange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.legend(loc='best')
    plt.show()
    
    return result_dict
