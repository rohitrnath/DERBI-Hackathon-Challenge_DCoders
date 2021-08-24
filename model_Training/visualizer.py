import os
import torch
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
import base64
import json
import copy

from gradcam import *

def deprocess(img, mean, std):
    img = img.permute(1,2,0)
    img = img * torch.Tensor(std) + torch.Tensor(mean)
    return img




def view_classify(img, ps, label, mean, std, heatmap=None):

    class_name = pathology_list
    classes = np.array(class_name)

    ps = ps.cpu().data.numpy().squeeze()
    img = deprocess(img, mean, std)
    #img = np.transpose(img, (1, 2, 0))[:,:,0]
    class_labels = list(np.where(label==1)[0])

    if not class_labels :
        title = 'No Findings'
    else : 
        title = itemgetter(*class_labels)(class_name)
        


    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,3), ncols=3)
    ax1.imshow(img)
    ax1.set_title('Ground Truth : {}'.format(title))
    ax1.axis('off')
    # if(heatmap != None):
    ax2.imshow(img)
    ax2.imshow(heatmap, alpha=0.25, cmap='jet')
    ax2.set_title('Gradcam')
    ax2.axis('off')

    ax3.barh(classes, ps)
    ax3.set_aspect(0.1)
    ax3.set_yticks(classes)
    ax3.set_yticklabels(classes)
    ax3.set_title('Predicted Class')
    ax3.set_xlim(0, 1.1)

    plt.tight_layout()

    return None