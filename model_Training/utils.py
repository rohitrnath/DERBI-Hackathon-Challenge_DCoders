import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from PIL import Image
import torch.nn as nn
from torchvision.utils import make_grid

def show_sample(img, target, invert=False):
  class_labels = list(np.where(target==1.0)[0])
  if invert:
      plt.imshow(np.transpose(1-img, (1, 2, 0))[:,:,2], cmap="Greys_r")
      #plt.imshow(1-img)
      #plt.imshow(1 - img.permute((1, 2, 0).squeeze()))
  else:
      plt.imshow(np.transpose(img, (1, 2, 0))[:,:,2], cmap="Greys_r")
      #plt.imshow(img)
      #plt.imshow(1 - img.permute((1, 2, 0).squeeze()))
  # print(img)
  plt.title(itemgetter(*class_labels)(pathology_list));


def show_batch(dl, invert=False):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(32, 64))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        img=make_grid(data, nrow=11)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))[:,:,2], interpolation='nearest', cmap="Greys_r")
        break

def plot_scores(history):
    scores = [x['val_score'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


def class_accuracy(dataloader, model, df):

    per_class_accuracy = [0 for i in range(len(pathology_list))]
    total = 0.0
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            ps = model(images.to(device))
            labels = labels.to(device)
            ps = (ps >= 0.5).float()

            for i in range(ps.shape[1]):
                x1 = ps[:,i:i+1]
                x2 = labels[:,i:i+1]
                per_class_accuracy[i] += int((x1 == x2).sum())

        per_class_accuracy = [(i/len(df))*100.0 for i in per_class_accuracy]

    return per_class_accuracy     


def get_acc_data(class_names,acc_list):
    df = pd.DataFrame(list(zip(class_names, acc_list)), columns =['Labels', 'Acc']) 
    return df


def roc_score(dataloader, model, df):

    outs = []
    labs = []
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            ps = model(images.to(device))
            labs.extend(labels.cpu().detach().numpy())
            outs.extend(ps.cpu().detach().numpy())
        return outs, labs
            
from sklearn.metrics import roc_auc_score, roc_curve
def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = np.asarray(generator)[:,i]
            pred = np.asarray(predicted_vals)[:,i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals