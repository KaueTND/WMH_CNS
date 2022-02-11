# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 23:31:16 2021

@author: kaueu
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import seaborn as sns
import pickle
import pandas as pd
import nibabel as nib
from sklearn.metrics import jaccard_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def space_it(total,title):
    total_char = total - len(title)    
    return ' '*(int(total_char/2))+title+' '*(int(total_char/2))
def f_measure(TN,FN,FP,TP):
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f_measure = (2*precision*recall)/(precision+recall)
    return f_measure

def iou(TN,FN,FP,TP):
    return TN

def get_lobe_step(lobe_volume,new,val_compare,val_input):
    lobe_volume = lobe_volume + (new == val_compare)*val_input 
    return lobe_volume

def get_lobe_label(wmparc,wmparc_confusion_matrix):
    lobe_volume = np.zeros((256*256*64))
    new = wmparc.reshape((256*256*64))    
    #wm-lh-frontal = [3002,3003,3012,3014,3017,3018,3019,3020,3024,3026,3027,3028,3032]
    #wm-lh-occipital = [3005,3011,3013,3021]    
    #wm-lh-temporal = [3006,3007,3009,3015,3016,3030,3033,3034]
    #wm-lh-parietal = [3008,3010,3022,3023,3025,3029,3031]
    #wm-rh-frontal = [4002,4003,4012,4014,4017,4018,4019,4020,4024,4026,4027,4028,4032]:
    #wm-rh-occipital = [4005,4011,4013,4021]
    #wm-rh-temporal = [4006,4007,4009,4015,4016,4030,4033,4034]
    #wm-rh-parietal = [4008,4010,4022,4023,4025,4029,4031]
    #wm-rh-insula = [4035]
    #wm-lh-insula = [3035]    
    lobe_labels = [[3002,3003,3012,3014,3017,3018,3019,3020,3024,3026,3027,3028,3032,4002,4003,4012,4014,4017,4018,4019,4020,4024,4026,4027,4028,4032],#frontal
                   [3005,3011,3013,3021,4005,4011,4013,4021],#occiptal
                   [3001,3006,3007,3009,3015,3016,3030,3033,3034,4001,4006,4007,4009,4015,4016,4030,4033,4034],#temporal
                   [3008,3010,3022,3023,3025,3029,3031,4008,4010,4022,4023,4025,4029,4031],#parietal
                   [3035,4035],#insula
                   ]        
    
    values_f_measure = list()
    values_N = list()
    for idx, regions in enumerate(lobe_labels):
        for region in regions:
            #print(region)
            lobe_volume = get_lobe_step(lobe_volume,new,region,idx+1)
        new_test_lobe = lobe_volume.reshape((256,256,64))
        #print(np.unique(new_test_lobe))
        #wm thresholding
        lobe_binary = new_test_lobe==(idx+1)
        wmparc_confusion_new = lobe_binary*(wmparc_confusion_matrix)
        #print(wmparc_confusion_matrix)
        #print(np.bincount(wmparc_confusion_new.ravel(),None,5)[1:])        
        [tn,fn,fp,tp] = np.bincount(wmparc_confusion_new.ravel(),None,5)[1:]
        #print()

        fmeasure = f_measure(tn,fn,fp,tp)
        fmeasure = fmeasure if fmeasure > 0 else 1
        values_f_measure.append(fmeasure)
        values_N.append(tp+fn)#fp,fn,tp
        #print(fmeasure)
        

    return np.array(values_f_measure), np.array(values_N)

def show_img_lobes(title,fmeasure,N):
    img = cv2.imread('viewLobe.png')
    #cv2.imshow('ds',img)    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontTitle = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2
       
    [frontal_f,occipital_f,temporal_f,parietal_f,insula_f] = ["%.2f" % x for x in fmeasure]
    [frontal_n,occipital_n,temporal_n,parietal_n,insula_n] = [int(x) for x in N]
    
    #
    #title='Resnet153'
    
    img = cv2.putText(img, space_it(14,title), (300, 100), font, 
                       1.4, color, 5)
    
    
    # Using cv2.putText() method to put accuracy
    #frontal
    img = cv2.putText(img, str(frontal_f)+'%', (110, 190), font, 
                       fontScale, color, 2, cv2.LINE_AA)
    #parietal
    img = cv2.putText(img, str(parietal_f)+'%', (640, 210), font, 
                       fontScale, color, 2, cv2.LINE_AA)
    #insula
    img = cv2.putText(img, str(insula_f)+'%', (120, 675), font, 
                       fontScale, color, 2, cv2.LINE_AA)
    #occipital
    img = cv2.putText(img, str(occipital_f)+'%', (640, 635), font, 
                       fontScale, color, 2, cv2.LINE_AA)
    #temporal
    img = cv2.putText(img, str(temporal_f)+'%', (390, 695), font, 
                       fontScale, color, 2, cv2.LINE_AA)

    #Showing the results
    img = cv2.putText(img, '[N='+str(frontal_n)+']', (110, 230), font, 
                       fontScale, color, 2, cv2.LINE_AA)

    img = cv2.putText(img, '[N='+str(parietal_n)+']', (640, 250), font, 
                       fontScale, color, 2, cv2.LINE_AA)

    img = cv2.putText(img, '[N='+str(insula_n)+']', (120, 715), font, 
                       fontScale, color, 2, cv2.LINE_AA)

    img = cv2.putText(img, '[N='+str(occipital_n)+']', (640, 675), font, 
                       fontScale, color, 2, cv2.LINE_AA)

    img = cv2.putText(img, '[N='+str(temporal_n)+']', (390, 735), font, 
                       fontScale, color, 2, cv2.LINE_AA)    

    return img

def get_volume_confusion_matrix(path,file,method=None):
    gt = nib.load(path+'true_mask/'+file+'.nii')
    gt = np.asarray(gt.get_fdata(),dtype='uint8')    
    #layer_gt = gt[:,:,sliceId].T

    predi_ = nib.load(path+'predicted_mask_'+method.lower()+'/'+file+'.nii')
    predi_ = np.asarray(predi_.get_fdata(),dtype='uint8')    
    output = gt + predi_*2
    return output

def plot_image_gt(path,file,sliceId,method=None):
    
    orig = nib.load(path+'flair/'+file+'.nii')
    orig = np.asarray(orig.get_fdata(),'int32')
    gt = nib.load(path+'true_mask/'+file+'.nii')
    gt = np.asarray(gt.get_fdata(),dtype='uint8')
    output = None    

    
    layer_gt = gt[:,:,sliceId].T
    layer_orig = orig[:,:,sliceId].T
    layer_orig = np.asarray(((layer_orig-layer_orig.min()) / (layer_orig.max()-layer_orig.min()))*255,dtype='int32')
    img_rgb = np.stack(((layer_orig,)*3),axis=-1)#np.zeros((h, w, 3), dtype=np.uint8)
    
    
    
    if method!=None:
        predi_ = nib.load(path+'predicted_mask_'+method.lower()+'/'+file+'.nii')
        predi_ = np.asarray(predi_.get_fdata(),dtype='uint8')    
        layer_predi_ = predi_[:,:,sliceId].T
        output = layer_gt + layer_predi_*2
        h, w = output.shape
        
        for gray, rgb in label_to_color.items():
            img_rgb[output == gray, :] = rgb  
    #img_rgb[layer_orig < 15, :] = [255,255,255]  
    
    #if method!=None:
    #    predi_   = np.rot90(gt[:,:,sliceId-1])#predi_ and without -1
    #    output = layer_gt + predi_*2
    #    h, w = output.shape
        
    #    for gray, rgb in label_to_color.items():
    #        img_rgb[output == gray, :] = rgb    
    return [img_rgb,output]






choice = 1

# 1 - Learning Rate
# 2 - Plotting Ground-truth
# 3 - Lobe Identification

if choice == 1:
    a = pickle.load(open('3D_model_vgg19_4000epochs_history.pkl','rb'))
    vgg19 = {'iou':a['acc'], 'f':a['f1-score'],'loss':a['loss']}
     
    a = pickle.load(open('3D_model_resnet152_4000epochs_history.pkl','rb'))
    resnet = {'iou':a['acc'], 'f':a['f1-score'],'loss':a['loss']}
    
    a = pickle.load(open('3D_model_vgg16_4000epochs_history.pkl','rb'))
    vgg16 = {'iou':a['acc'], 'f':a['f1-score'],'loss':a['loss']}     

    a = pickle.load(open('3D_model_efficientnetb0_4000epochs_history.pkl','rb'))
    efficientnetb0 = {'iou':a['acc'], 'f':a['f1-score'],'loss':a['loss']}     
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,2    ))
    epochs = range(1, len(vgg19['loss']) + 1)
    sns.lineplot(x=epochs, y=vgg19['iou'],label='VGG19') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=resnet['iou'],label='ResNet') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=vgg16['iou'],label='VGG16') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=efficientnetb0['iou'],label='EfficientNetB0') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=vgg19['f'], label='sadsdsad')
    plt.title('Training curves vs Number of epochs')
    plt.xlabel('Epochs')
    ax = plt.gca()
    ax.set_ylim(0,1)
    plt.ylabel('IoU')
    plt.legend()
    #plt.show()
    plt.savefig("IoU_learning.eps",bbox_inches='tight')
    
    plt.figure(figsize=(12,2    ))
    epochs = range(1, len(vgg19['loss']) + 1)
    sns.lineplot(x=epochs, y=vgg19['f'],label='VGG19') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=resnet['f'],label='ResNet') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=vgg16['f'],label='VGG16') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=efficientnetb0['f'],label='EfficientNetB0') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=vgg19['f'], label='sadsdsad')
    plt.title('Training curves vs Number of epochs')
    plt.xlabel('Epochs')
    #plt.set_ylim([0,2])
    ax = plt.gca()
    ax.set_ylim(0,1)
    plt.ylabel('F1-score')
    plt.legend()
    #plt.show()
    plt.savefig("F1-score_learning.eps",bbox_inches='tight')
    
    plt.figure(figsize=(12,2    ))
    epochs = range(1, len(vgg19['loss']) + 1)
    sns.lineplot(x=epochs, y=vgg19['loss'],label='VGG19') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=resnet['loss'],label='ResNet') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=vgg16['loss'],label='VGG16') #'y', label='Training loss')
    sns.lineplot(x=epochs, y=efficientnetb0['loss'],label='EfficientNetB0') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=vgg19['f'], label='sadsdsad')
    plt.title('Training curves vs Number of epochs')
    plt.xlabel('Epochs')
    #plt.set_ylim([0,2])
    ax = plt.gca()
    ax.set_ylim(0,1)
    plt.ylabel('Dice+Focal loss')
    plt.legend()
    #plt.show()
    plt.savefig("Loss_learning.eps",bbox_inches='tight')
elif choice == 2:
    sns.set_style("white")
    total = 6
    
    #ax = [plt.subplot(4,6,i+1) for i in range(24)]
    
    label = ['Original','VGG16','VGG19','Resnet152']
    methods = [None,'vgg16','vgg19','resnet152']
    files = [['NORM325_AA',29],
             ['NORM456_MC',28],
             ['NORM466_GR',31],
             ['NORM497_BO',26],
             ['NORM640_DM',23],
             ['NORM675_SS',27]]
    
    
    label_to_color = {
    #0: [0,  0, 0],
    1: [255, 0  , 0  ],
    2: [0  , 255, 0  ],
    3: [0  , 0  , 255]
    }
    

    #plt.figure(figsize=(16, 8))
    #plt.subplot(1, 2, 1)#,# plt.imshow(output)
    #plt.imshow(img_rgb)
    #      
    
   
    
    fig, axs = plt.subplots(4, total,figsize=(22,15))#
    for i in range(4):
        for j in range(total):      
            if j == 0:
                axs[i,j].set_ylabel(label[i],labelpad=-5)
            axs[i,j].imshow( plot_image_gt('../processed_flair/',files[j][0],files[j][1],methods[i]),origin='lower')
            
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
        
    
    #
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("brainError.eps",bbox_inches='tight')

elif choice==3: 
    
    sns.set_style("white")
    total = 3
    
    #ax = [plt.subplot(4,6,i+1) for i in range(24)]
    
    label = ['Original','VGG16','VGG19','Resnet152','EfficientNetB0']
    methods = [None,'vgg16','vgg19','resnet152','efficientnetb0']
    files = [['NORM325_AA',29],
             ['NORM456_MC',28],
    #         ['NORM466_GR',31],
     #        ['NORM497_BO',26],
      #       ['NORM640_DM',23],
             ['NORM675_SS',27]]
    
    
    label_to_color = {
    #0: [0,  0, 0],
    1: [255, 0  , 0  ],
    2: [0  , 255, 0  ],
    3: [255  , 255  , 0]
    }
    

    #plt.figure(figsize=(16, 8))
    #plt.subplot(1, 2, 1)#,# plt.imshow(output)
    #plt.imshow(img_rgb)
    #      
    
   
    
    fig, axs = plt.subplots(total, 5,figsize=(24.3,15))#
    for i in range(total):
        for j in range(5):      
            if i == 0:
                axs[i,j].set_xlabel(label[j],labelpad=0,fontsize=20)
                #axs[i,j].xaxis.tick_top()
                axs[i,j].xaxis.set_label_coords(0.5, 1.1)
            img = plot_image_gt('../processed_flair/',files[i][0],files[i][1],methods[j])[0]
            axs[i,j].imshow(img[75:175,80:180,:] ,origin='lower')
            
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
        
    
    rp = mpatches.Patch(color='red', label='False Negative')
    gp = mpatches.Patch(color='green', label='False Positive')
    yp = mpatches.Patch(color='yellow', label='True Positive')
    plt.legend(handles=[rp,gp,yp],ncol=3,loc='lower left', bbox_to_anchor=(-2.8, -0.2, 0.5, 0.5))
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("brainError.eps",bbox_inches='tight')
elif choice==4:
    
    files = pd.read_csv('FLAIR_subjects.txt').to_numpy()
    full_path = '../processed_flair/' 
    aaa = list()
    vgg16 = list()
    vgg19 = list()
    efficientnetb0 = list()
    total = list()
    
    methods = ['VGG16','VGG19','Resnet152','EfficientNetB0']
    
    method_dict = {"VGG16":list(),"VGG19":list(),"Resnet152":list(),"EfficientNetB0":list()}
    
    for file in files:
       
        file = file[0]
        print(file)
        nib_file_wmparc = nib.load(full_path+'registered_wmparc/flair_'+file+'.wmparc.nii')
        
        
        wmparc_l = nib_file_wmparc.get_fdata() 
        wm_binary = wmparc_l>1
        for method in methods:

            confusion_matrix = get_volume_confusion_matrix('../processed_flair/',file,method)

            
            wmparc_confusion_matrix = wm_binary*(confusion_matrix+1)
                        
            [tn,fn,fp,tp] = np.bincount(wmparc_confusion_matrix.ravel())[1:]
            
            method_dict[method].append(f_measure(tn,fn,fp,tp))#[tn,fn,fp,tp])
        total.append('[N='+str(tp+fn)+']')
    total = np.array(total)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,12))
    epochs = range(files.shape[0])
    sns.scatterplot(x=epochs, y=method_dict['VGG19'],label='VGG19',marker="^",s=300) #'y', label='Training loss')
    sns.scatterplot(x=epochs, y=method_dict['Resnet152'],label='ResNet',marker='s',s=300) #'y', label='Training loss')
    sns.scatterplot(x=epochs, y=method_dict['VGG16'],label='VGG16',marker='o',s=300) #'y', label='Training loss')
    sns.scatterplot(x=epochs, y=method_dict['EfficientNetB0'],label='EfficientNetB0',marker='X',s=300) #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=vgg19['f'], label='sadsdsad')
    #plt.title('F-Measure')
    plt.xlabel('Subjects ID',fontsize=22)
    
    ax = plt.gca()
    plt.rcParams.update({'font.size':22})
    ax.set_ylim(0.4,1)
    #ax.set_xlim(-12,20)
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels( epochs)
    plt.ylabel('F-Measure')
    plt.legend()
    
    for i in epochs:
        plt.text(i,1.01,total[i],rotation=45)
    
    #plt.show()
    plt.savefig("f_measure_subjects.eps",bbox_inches='tight')

elif choice == 5:
    
    files = pd.read_csv('FLAIR_subjects.txt').to_numpy()
    full_path = '../processed_flair/' 
    aaa = list()
    vgg16 = list()
    vgg19 = list()
    efficientnetb0 = list()
    
    methods = ['VGG16','VGG19','Resnet152','EfficientNetB0']
    
    method_dict = {"VGG16":list(),"VGG19":list(),"Resnet152":list(),"EfficientNetB0":list()}
    
    for method in methods:    
        table_fmeasure_total = np.zeros((20,5))
        table_N_total = np.zeros((20,5))
        
        for idx,file in enumerate(files):
       
            file = file[0]
            #print(file)
            nib_file_wmparc = nib.load(full_path+'registered_wmparc/flair_'+file+'.wmparc.nii')
            
            
            wmparc_l = nib_file_wmparc.get_fdata() 
            wm_binary = wmparc_l>3000
            
    
            confusion_matrix = get_volume_confusion_matrix('../processed_flair/',file,method)    
            wmparc_confusion_matrix = wm_binary*(confusion_matrix+1)
            table_fmeasure_total[idx,:],table_N_total[idx,:] = get_lobe_label(wmparc_l,wmparc_confusion_matrix)
        print(method)
        #frontal,occipital,temporal,parietal,insula
        ffmeasure = np.mean(table_fmeasure_total,axis=0)*100
        print(ffmeasure)
        fN = np.sum(table_N_total,axis=0)
        print(fN)
        
        img = show_img_lobes(method,ffmeasure,fN)
        cv2.imwrite('lobes_'+method+'.png',img)
        #cv2.imshow('ds',img)
        #if cv2.waitKey(0) == 27:
        #    ok_flag = False
    
        #cv2.destroyAllWindows()         
        
        #break

elif choice == 6:
    
    img = cv2.imread('viewLobe.png')
    cv2.imshow('ds',img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontTitle = cv2.FONT_HERSHEY_SIMPLEX
      
    # fontScale
    fontScale = 1
       
    # Blue color in BGR
    color = (0, 0, 0)
      
    # Line thickness of 2 px
    thickness = 2
       
    
    
    #
    #title=
    
    #img = cv2.putText(img, space_it(14,title), (300, 100), font, 
     #                  1.4, color, 5)
    
    img = show_img_lobes('Resnet153',99.99,99.99,99.99,99.99,99.99,400,400,400,400,400)
    # Using cv2.putText() method to put accuracy
    # img = cv2.putText(img, str(99.99)+'%', (110, 190), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, str(99.99)+'%', (640, 210), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, str(99.99)+'%', (120, 675), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, str(99.99)+'%', (640, 635), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, str(100.99)+'%', (390, 695), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # #Showing the results
    # img = cv2.putText(img, '[N='+str(400)+']', (110, 230), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, '[N='+str(400)+']', (640, 250), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, '[N='+str(400)+']', (120, 715), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, '[N='+str(400)+']', (640, 675), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)

    # img = cv2.putText(img, '[N='+str(400)+']', (390, 735), font, 
    #                    fontScale, color, 2, cv2.LINE_AA)    





    cv2.imshow('ds',img)
    if cv2.waitKey(0) == 27:
        ok_flag = False

    cv2.destroyAllWindows()                        
            
elif choice == 7:
    sns.set_style("white")
    total = 3
    
    #ax = [plt.subplot(4,6,i+1) for i in range(24)]
    
    methods = ['VGG16','VGG19','Resnet152','EfficientNetB0']
    files = pd.read_csv('FLAIR_subjects.txt').to_numpy()
    method_out = {"VGG16":np.zeros((5,len(files))), 
                   "VGG19":np.zeros((5,len(files))),
                   "Resnet152":np.zeros((5,len(files))),
                   "EfficientNetB0":np.zeros((5,len(files)))
                   }
    
    metricHaus = np.zeros((4,len(files)))    
    
    for idx, method in enumerate(methods):
        print(method)
        metricHaus[idx,:] = np.load(method.lower()+'_props.npy')[1,:]
        
   
    full_path = '../processed_flair/' 
    aaa = list()
    vgg16 = list()
    vgg19 = list()
    efficientnetb0 = list()
    total = list()
    
   
    method_dict = {"VGG16":list(),"VGG19":list(),"Resnet152":list(),"EfficientNetB0":list()}
    
    for file in files:
       
        file = file[0]
        print(file)
        nib_file_wmparc = nib.load(full_path+'registered_wmparc/flair_'+file+'.wmparc.nii')
        
        
        wmparc_l = nib_file_wmparc.get_fdata() 
        wm_binary = wmparc_l>1
        for method in methods:

            confusion_matrix = get_volume_confusion_matrix('../processed_flair/',file,method)

            
            wmparc_confusion_matrix = wm_binary*(confusion_matrix+1)
                        
            [tn,fn,fp,tp] = np.bincount(wmparc_confusion_matrix.ravel())[1:]
            
            method_dict[method].append(f_measure(tn,fn,fp,tp))#[tn,fn,fp,tp])
        total.append('[N='+str(tp+fn)+']')
    total = np.array(total)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,12))
    epochs = range(files.shape[0])
    
    sns.scatterplot(x=epochs, y=metricHaus[1,:],label='VGG19',marker="^",s=300) #'y', label='Training loss')    
    sns.scatterplot(x=epochs, y=metricHaus[2,:],label='ResNet',marker='s',s=300) #'y', label='Training loss')    
    sns.scatterplot(x=epochs, y=metricHaus[0,:],label='VGG16',marker='o',s=300) #'y', label='Training loss')
    sns.scatterplot(x=epochs, y=metricHaus[3,:],label='EfficientNetB0',marker='X',s=300) #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=vgg19['f'], label='sadsdsad')
    #plt.title('F-Measure')
    plt.xlabel('Subjects ID',fontsize=22)
    
    ax = plt.gca()
    plt.rcParams.update({'font.size':22})
    ax.set_ylim(-0.5,50)
    #ax.set_xlim(-12,20)
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels( epochs)
    plt.ylabel('Hausdorff Distance')
    plt.legend()
    
    for i in epochs:
        plt.text(i,51.01,total[i],rotation=45)
    
    #plt.show()
    plt.savefig("f_measure_Hausdorff.eps",bbox_inches='tight')        
    
    
    
    
    
    
    #sns.set_style("whitegrid")
    #plt.figure(figsize=(12,4))
    #epochs = range(files.shape[0])
    #sns.lineplot(x=epochs, y=method_dict['VGG19'],label='VGG19') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=method_dict['Resnet152'],label='ResNet') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=method_dict['VGG16'],label='VGG16') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=method_dict['EfficientNetB0'],label='EfficientNetB0') #'y', label='Training loss')
    #sns.lineplot(x=epochs, y=vgg19['f'], label='sadsdsad')
    #plt.title('F-Measure')

        
    #set three-channels for the image
    
    #iterate over each image
    #set the color of the ground-truth



    
    #teste
  
    # file_Nifti  = nib.Nifti1Image(orig, affine=np.eye(4,4))
    # nib.save(file_Nifti,full_path+'orig_test.nii')    

    # wparc1 = wmparc_l>0
    # wparc2 = wmparc_l>3000
    # wpartf = wparc1*1+wparc2*1

    # file_Nifti  = nib.Nifti1Image(wpartf.astype('float32'), affine=np.eye(4,4))
    # nib.save(file_Nifti,full_path+'wmparc_test.nii')         

    # file_Nifti  = nib.Nifti1Image(gtmask, affine=np.eye(4,4))
    # nib.save(file_Nifti,full_path+'gtmask_test.nii')    



    
    #break
    
    
    #vol_img = nib_file_img.get_fdata()
    
    #vol_img = (resize(vol_img, (256,256,64),anti_aliasing=True))
    #vol_img = np.asanyarray(vol_img,int)
    #print(vol_img.shape)
    #np.save('../processed_images/img/'+file,vol_img)

    #vol_mask = nib_file_mask.get_fdata()
    
    #vol_mask = (resize(vol_mask, (256,256,64),anti_aliasing=True))
    #vol_mask = np.asanyarray(vol_mask,int)
    #print(vol_img.shape)
    #np.save('../processed_images/mask/'+file,vol_mask)
    #break



    
    
#print(history.history)

# plt.figure(figsize=(12,8))
# plt.plot(epochs, acc, 'y', label='Training IoU')
# np.save('iou_train',acc)
# plt.plot(epochs, f, 'b', label='Training F1-score')
# np.save('f1_train',f)
# plt.title('Training score')
# plt.xlabel('Epochs')
# plt.ylabel('effectiveness')
# plt.legend()
# plt.show()






# values = np.load('valor_por_paciente.npy').T# tn,fn,fp,tp
# tn = values[0,:]
# fn = values[1,:]
# fp = values[2,:]
# tp = values[3,:]
# plt.xlabel('Subjects')
# plt.ylabel('F-measure')
# 110 / 110 +2
# precision = tp / (tp+fp)
# recall = tp / (tp+fn)
# fmeasure = (2*precision*recall)/(precision+recall)
# x = np.arange(1,20)
# sns.barplot(x=x,y=fmeasure)



# var = [[71777,0,0,0],
# [18406,0,2,46],
# [22263,0,0,0],
# [45270,0,0,0],
# [8692,0,0,0],
# [95621,0,0,3],
# [25716,4,9,28],
# [35557,0,0,0],
# [60605,1,0,23],
# [11265,0,0,0],
# [86874,2,0,0],
# [29565,0,0,23],
# [30294,0,0,0],
# [55597,0,0,0],
# [9415,0,0,0],
# [76536,0,0,0],
# [18790,0,0,0],
# [27859,0,0,0],
# [47950,0,0,0],
# [9214,0,0,0],
# [89144,0,0,0],
# [22703,2,2,41],
# [25813,0,0,0],
# [53801,0,0,0],
# [10319,0,0,0],
# [75753,1,0,7],
# [21230,4,0,37],
# [27082,0,0,0],
# [51425,1,0,22],
# [8716,3,0,0],
# [66201,0,0,0],
# [17554,1,0,0],
# [20875,0,0,0],
# [45039,0,0,0],
# [8200,0,0,0],
# [73649,16,0,0],
# [18586,1,0,2],
# [22532,0,0,0],
# [46120,0,2,2],
# [8525,0,0,0],
# [72685,0,0,0],
# [21647,0,0,0],
# [27306,0,0,0],
# [48121,0,0,0],
# [8218,0,0,0],
# [87152,0,0,0],
# [23622,0,6,2],
# [29192,0,0,0],
# [55277,0,0,0],
# [9855,0,0,0],
# [79472,0,0,0],
# [22736,14,5,14],
# [25936,0,0,0],
# [52227,0,0,0],
# [8259,0,0,0],
# [61884,0,0,0],
# [18399,2,0,16],
# [22314,0,0,0],
# [45680,0,0,0],
# [7969,0,0,0],
# [62593,0,0,0],
# [17344,1,0,0],
# [19683,0,0,0],
# [44689,0,0,0],
# [8420,0,0,0],
# [89212,0,0,0],
# [26858,0,0,0],
# [30351,0,0,0],
# [55203,0,0,0],
# [10144,0,0,0],
# [86649,0,0,0],
# [27188,1,0,31],
# [31236,0,0,0],
# [57656,0,0,0],
# [9285,0,0,0],
# [80679,0,0,0],
# [27141,1,0,30],
# [27245,0,0,0],
# [52408,0,0,0],
# [9441,0,0,0],
# [74680,0,0,0],
# [19958,0,0,0],
# [24120,0,0,0],
# [48688,0,0,0],
# [9304,0,0,0],
# [70098,0,0,0],
# [18746,9,1,43],
# [25538,0,0,0],
# [50214,0,0,0],
# [9914,0,0,0],
# [80592,4,0,0],
# [16183,0,0,0],
# [27966,0,0,0],
# [53523,0,0,4],
# [9514,2,0,0]]

# frontal = list()
# occipital = list()
# parietal = list()
# temporal = list()
# insula = list()

# for idx,valor in enumerate(var):
#     if idx%5 == 0:
#         frontal.append(valor)
#     elif idx%5 == 1:
#         occipital.append(valor)
#     elif idx%5 == 2:
#         temporal.append(valor)
#     elif idx%5 == 3:
#         parietal.append(valor)
#     elif idx%5 == 4:
#         insula.append(valor)

# frontal = np.asarray(frontal)
# occipital = np.asarray(occipital)
# temporal = np.asarray(temporal)
# parietal = np.asarray(parietal)
# insula = np.asarray(insula)

# [tn,fn,fp,tp] = np.sum(parietal.T,axis=1)

# acc = (tp+tn)/(tp+tn+fp+fn)





