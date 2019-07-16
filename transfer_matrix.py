from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# y_true = [2,1,0,1,1,3,1,0]
# y_pred = [2,0,0,1,2,3,2,1]
#
# cm = confusion_matrix(y_true,y_pred)
tm = np.matrix([[np.nan,2,1.8,0.3],[2,np.nan,2,1],[1.8,2,np.nan,0.2],[0,1.2,1,np.nan]])
print(tm)

def plot_confusion_matrix(cm, labels_name,title):
    plt.imshow(cm)#,interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local,labels_name,rotation=90)
    plt.yticks(num_local,labels_name)
    plt.ylabel('Original Scenario')
    plt.xlabel('New Scenario')

labels_name=['Original','Zoom','Noisy','HV-Flip']
plot_confusion_matrix(tm,labels_name,'Transfer Matrix')
plt.savefig('./Transfermatrix.png',format='png',bbox_inches='tight')
#tight parameter used to save the whole image
plt.show()
