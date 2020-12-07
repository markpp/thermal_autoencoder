import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(dataset='test_abnormal'):
    data_dir = os.path.join('output',dataset)
    losses = np.load(os.path.join(data_dir,'losses.npy'))

    #images = np.load(os.path.join(data_dir,'inputs.npy'))
    #recs = np.load(os.path.join(data_dir,'recs.npy'))
    names = np.load(os.path.join(data_dir,'files.npy'))
    #names = [os.path.basename(n)[:-4] for n in names]
    if 'abnormal' in dataset:
        labels = [1]*losses.shape[0]
    else:
        labels = [0]*losses.shape[0]
    return losses, np.array(labels), np.array(names)

if __name__ == "__main__":

    data_type = 'test'
    datasets = ['{}_normal'.format(data_type),'{}_abnormal'.format(data_type)]
    #datasets = ['{}_normal'.format(data_type),'{}_abnormal'.format(data_type),'{}_entering'.format(data_type)]

    losses, labels, names = np.array([]), np.array([]), np.array([])

    plt.figure(figsize=(8,6))
    #plt.xlabel('random', fontsize=14)
    plt.ylabel('reconstruction error (MSE)', fontsize=12)

    file = open(os.path.join('output','extreems_{}.txt'.format(data_type)),"w")

    for dataset in datasets:
        loss, lab, name = load_data(dataset=dataset)

        # find extreems
        indices = np.argsort(loss)
        file.write("{}\n".format(dataset))
        #print(indices[-3:])
        for l, n in zip(loss[indices[:3]],name[indices[:3]]):
            file.write("loss {}, file {}\n".format(l, n))
        middle_idx = len(indices)//2
        for l, n in zip(loss[indices[middle_idx-1:middle_idx+1]],name[indices[middle_idx-1:middle_idx+1]]):
            file.write("loss {}, file {}\n".format(l, n))
        for l, n in zip(loss[indices[-3:]],name[indices[-3:]]):
            file.write("loss {}, file {}\n".format(l, n))

        x = np.random.random_sample(loss.shape[0])
        if 'abnormal' in dataset:
            plt.plot(x,loss,'.',color='r', label='Abnormal')
        elif 'entering' in dataset:
            plt.plot(x,loss,'.',color='b', label='entering')
        else:
            plt.plot(x,loss,'.',color='g', label='Normal')

        losses = np.concatenate((losses,loss), axis=0)
        labels = np.concatenate((labels,lab), axis=0)
        names = np.concatenate((names,name), axis=0)

    file.close()

    y_score = losses
    y_true = labels


    from sklearn.metrics import precision_recall_curve
    # calculate model precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    desired_recall = 0.8
    idx = np.abs(recall - desired_recall).argmin()
    threshold = thresholds[idx]
    threshold = 0.0029042831156402826
    print('threshold: {}'.format(threshold))
    y_pred = np.where(y_score > threshold, 1, 0)


    plt.axhline(y=threshold,color='r',linestyle='--')

    ax = plt.subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output','reconstruction_error_{}.png'.format(data_type)))
    plt.clf()

    from sklearn import metrics
    auc_pr = metrics.auc(recall, precision)
    # plot the model precision-recall curve
    plt.plot(recall, precision, marker='.', label='AUC: {}'.format(auc_pr))

    plt.axvline(x=recall[idx],color='r',linestyle='--')

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ax = plt.subplot(111)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output','precision_recall_curve_{}.png'.format(data_type)))
    plt.clf()


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    res = "tn {}, fp {}, fn {}, tp {}\n".format(tn, fp, fn, tp)
    print(res)
    precision = tp / (tp+fp) 
    recall = tp / (tp+fn)
    pr = "recall: {:.3f}, precision: {:.3f}".format(recall,precision)
    print(pr)
    file = open(os.path.join('output','results_{}.txt'.format(data_type)),"w")
    file.write("threshold: {}\n".format(threshold))
    file.write(res)
    file.write(pr)
    file.close()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    import seaborn as sn
    import pandas as pd

    df_cm = pd.DataFrame(cm, index = ["Normal", "Abnormal"],
                             columns = ["Predicted Normal", "Predicted Abnormal"])
    plt.figure(figsize = (8,6))
    sn.heatmap(df_cm, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join('output','confusion_matrix_{}.png'.format(data_type)))

    '''
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score, average='weighted')

    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
    '''
