from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from munkres import Munkres
import random
import warnings

# warnings.filterwarnings('ignore')

def TestClassifacation(embedding, label):

    method = SVC(kernel="linear", max_iter=90000)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    # if
    n_scores = cross_val_score(
        method, embedding, label, scoring="accuracy", cv=cv, n_jobs=-1
    )

    return (
        n_scores.mean(),
        n_scores.std(),
    )

def TestClassifacationLogisticRegression(embedding, label):

    # method = KNeighborsClassifier()
    scaler = StandardScaler()
    embedding = scaler.fit_transform(embedding)

    # method = LogisticRegression(C=0.5,solver='lbfgs',multi_class='auto')
    
    num = embedding.shape[0]
    num_calss = np.max(label) + 1
    
    num_train = 20*num_calss
    num_val = 30*num_calss + num_train
    
    val_list = []
    test_list = []
    
    for i in range(20):
        random.seed(i)
        randomindex = random.sample(range(num), num)

        train_X = embedding[randomindex[:num_train], :]
        train_Y = label[randomindex[:num_train]]
        val_X = embedding[randomindex[num_train:num_val], :]
        val_Y = label[randomindex[num_train:num_val]]
        test_X = embedding[randomindex[num_val:], :]
        test_Y = label[randomindex[num_val:]]

        method = LogisticRegression(max_iter=1000, C=1, solver='lbfgs', multi_class='auto')
        method.fit(train_X, train_Y)
        
        # print(method.predict(train_X))

        train_acc = metrics.accuracy_score( method.predict(train_X), train_Y )
        val_acc = metrics.accuracy_score( method.predict(val_X), val_Y )
        test_acc = metrics.accuracy_score( method.predict(test_X), test_Y )
        
        val_list.append(val_acc)
        test_list.append(test_acc)

        # print('train', train_acc)
        # print('val', val_acc)
        # print('test', test_acc)

    val_mean = np.mean(val_list)
    val_std = np.std(val_list)
    test_mean = np.mean(test_list)
    test_std = np.std(test_list)

    print('C = {} TestClassifacationLogisticRegression--->--->'.format(1) ,val_mean, val_std, test_mean, test_std)

    return val_mean, val_std, test_mean, test_std

    # return (
    #     n_scores.mean(),
    #     n_scores.std(),
    # )

def TestClassifacationKNN(embedding, label):

    method = KNeighborsClassifier(n_neighbors=1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    # if
    n_scores = cross_val_score(
        method, embedding, label, scoring="accuracy", cv=cv, n_jobs=-1
    )

    return (
        n_scores.mean(),
        n_scores.std(),
    )



def TestClassifacationnmi(embedding, label):

    method = KNeighborsClassifier(n_neighbors=1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    # if
    n_scores = cross_val_score(
        method, embedding, label, scoring="accuracy", cv=cv, n_jobs=-1
    )

    return (
        n_scores.mean(),
        n_scores.std(),
    )

def TestClassifacationSPE(embedding, label):



    l1 = list(set(label))
    numclass1 = len(l1)
    method = SpectralClustering(n_clusters=numclass1, random_state=0, n_jobs=-1)

    # f_adj = np.matmul(embedding, np.transpose(embedding))
    predict_labels = method.fit_predict(embedding)

    # predict_labels = method.fit_predict(embedding)

    l2 = list(set(predict_labels))
    numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if predict_labels[i1] == c2]

            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(predict_labels))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(predict_labels) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(label, new_predict)
    f1_macro = metrics.f1_score(label, new_predict, average='macro')
    precision_macro = metrics.precision_score(label, new_predict, average='macro')
    # recall_macro = metrics.recall_score(label, new_predict, average='macro')
    # f1_micro = metrics.f1_score(label, new_predict, average='micro')
    # precision_micro = metrics.precision_score(label, new_predict, average='micro')
    # recall_micro = metrics.recall_score(label, new_predict, average='micro')

    nmi=metrics.normalized_mutual_info_score(label, predict_labels)
    adjscore = metrics.adjusted_rand_score(label, predict_labels)

    print('acc, nmi, f1_macro, precision_macro, adjscore')
    print(acc, nmi, f1_macro, precision_macro, adjscore)
    return acc, nmi, f1_macro, precision_macro, adjscore


def TestClassifacationKMeans(embedding, label):

    l1 = list(set(label))
    numclass1 = len(l1)

    predict_labels = KMeans(n_clusters=numclass1, random_state=0).fit_predict(embedding)
    # predict_labels = method.predict(embedding)

    l2 = list(set(predict_labels))
    numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if predict_labels[i1] == c2]

            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(predict_labels))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(predict_labels) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(label, new_predict)
    f1_macro, precision_macro, nmi, adjscore = 0, 0, 0, 0
    f1_macro = metrics.f1_score(label, new_predict, average='macro')
    precision_macro = metrics.precision_score(label, new_predict, average='macro')
    # nmi=metrics.normalized_mutual_info_score(label, predict_labels)
    nmi = metrics.v_measure_score(label, predict_labels)
    # print(nmi, nmi2)
    adjscore = metrics.adjusted_rand_score(label, predict_labels)

    print('acc:{}, nmi:{}, f1_macro:{}, precision_macro:{}, adjscore:{}'.format(
        acc, nmi, f1_macro, precision_macro, adjscore))
    return acc, nmi, f1_macro, precision_macro, adjscore
