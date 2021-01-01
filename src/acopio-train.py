from __future__ import print_function

import torch
import torch.utils as utils
import torchvision
import numpy as np
import argparse
import pandas as pd
import random

from torch import nn
from torch import backends
from datetime import datetime

from torch.utils.data import Dataset
import torch.utils as utils
import torch.nn.functional as F


import time
from io import StringIO
import shutil

import os
from os import environ
import math
import csv
from csv import DictReader
import datetime
import argparse
import logging
import sys
from typing import Optional

import sklearn
from sklearn import preprocessing

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
np.random.seed(1)

torch.backends.cudnn.enabled = False

vars = {
    'zero-date': '0000-00-00',
    'remove-cols': ['ERZET','ARKTX','VBELN','POSNR','MATNR','EAN11','VGBEL','VGPOS','AENAM'],
    'onehot-cols': ['WERKS','MTART','SUMBD','KALSM','KOSTA','MATKL','BWART','MTVFP','ANZPK', 'KUNNR','ERNAM'],
    'onehot-vals': ['LFIMG', 'LGMNG', 'MBDAT_MONTH', 'AEDAT_MONTH', 'ERDAT_MONTH', 'WADAT_MONTH', 'LDDAT_MONTH', 'TDDAT_MONTH', 'LFDAT_MONTH', 'KODAT_MONTH', 'BLDAT_MONTH', 'WADAT_IST_MONTH', 'FKDAT_MONTH', 'WERKS_T300', 'MTART_ZMOD', 'MTART_ZPER', 'SUMBD_B', 'KALSM_ZTWE01', 'KOSTA_C', 'MATKL_20202020', 'MATKL_20202030', 'MATKL_20202040', 'MATKL_20202050', 'MATKL_20203010', 'MATKL_20203020', 'MATKL_20203031', 'MATKL_20203032', 'MATKL_20203033', 'MATKL_20203040', 'MATKL_20203051', 'MATKL_20203052', 'MATKL_20204000', 'MATKL_20206010', 'MATKL_20206020', 'MATKL_20304020', 'MATKL_20304030', 'MATKL_20304040', 'MATKL_20304050', 'MATKL_20304060', 'MATKL_20304070', 'MATKL_20304082', 'MATKL_20304083', 'MATKL_20304084', 'MATKL_20304085', 'MATKL_20304092', 'MATKL_20304093', 'MATKL_20304094', 'MATKL_20304095', 'MATKL_20304096', 'MATKL_20304097', 'MATKL_20304098', 'MATKL_20305020', 'MATKL_20305030', 'MATKL_20305040', 'MATKL_20305050', 'MATKL_20305060', 'MATKL_20305070', 'MATKL_20305080', 'MATKL_20305090', 'MATKL_20306020', 'MATKL_20306030', 'MATKL_20306040', 'MATKL_20306050', 'MATKL_20306062', 'MATKL_20306063', 'MATKL_20306064', 'MATKL_20306070', 'MATKL_20306080', 'MATKL_20306092', 'MATKL_20306093', 'MATKL_20306094', 'MATKL_20307020', 'MATKL_20307030', 'MATKL_20307042', 'MATKL_20307043', 'MATKL_20308000', 'MATKL_20400000', 'MATKL_20500000', 'MATKL_20701000', 'MATKL_20702000', 'MATKL_20704000', 'MATKL_20801000', 'MATKL_20802000', 'MATKL_20803000', 'MATKL_20804000', 'MATKL_30305000', 'MATKL_30401000', 'MATKL_30402000', 'MATKL_30403000', 'MATKL_30405000', 'MATKL_30406000', 'MATKL_30406010', 'MATKL_30406020', 'MATKL_30406030', 'MATKL_30407000', 'MATKL_30408000', 'MATKL_30409000', 'MATKL_30410000', 'MATKL_30602000', 'MATKL_30602020', 'MATKL_30602050', 'MATKL_30602080', 'MATKL_30602090', 'MATKL_30603000', 'MATKL_30603010', 'MATKL_30604000', 'MATKL_30609000', 'MATKL_30610000', 'MATKL_30611000', 'MATKL_30612000', 'MATKL_30613000', 'MATKL_30614000', 'MATKL_30615000', 'MATKL_30619000', 'MATKL_30620000', 'MATKL_30900000', 'MATKL_30906000', 'MATKL_31101000', 'MATKL_31103000', 'MATKL_31105000', 'MATKL_31106000', 'MATKL_31107000', 'MATKL_31108000', 'MATKL_31109000', 'MATKL_31111000', 'MATKL_31112000', 'MATKL_31113000', 'MATKL_31116000', 'MATKL_31117000', 'MATKL_31119000', 'MATKL_31120000', 'MATKL_31126000', 'MATKL_31129000', 'MATKL_31134000', 'MATKL_31135000', 'MATKL_31138000', 'MATKL_31139000', 'MATKL_31141000', 'MATKL_31148000', 'MATKL_31149000', 'MATKL_31151000', 'MATKL_31152000', 'MATKL_31154000', 'MATKL_31155000', 'MATKL_31157000', 'MATKL_31158000', 'MATKL_31160000', 'MATKL_31401000', 'MATKL_31404000', 'MATKL_31405000', 'MATKL_31406000', 'MATKL_31407000', 'MATKL_31408000', 'MATKL_31409000', 'MATKL_31410000', 'MATKL_31411000', 'MATKL_31412000', 'MATKL_31413000', 'MATKL_31414000', 'MATKL_31415000', 'MATKL_31416000', 'MATKL_50200000', 'MATKL_60101000', 'MATKL_60102000', 'MATKL_60104000', 'MATKL_60202000', 'MATKL_80900000', 'BWART_647', 'MTVFP_2', 'ANZPK_1', 'KUNNR_T300', 'ERNAM_ARODRIGUEZ', 'ERNAM_BBEOTEGUI', 'ERNAM_EALMEIDA', 'ERNAM_IPEREZ', 'ERNAM_SRODRIGUEZ', 'ERNAM_SSANCHEZ', 'ERNAM_T001', 'ERNAM_T001GT', 'ERNAM_T002', 'ERNAM_T002GT', 'ERNAM_T004', 'ERNAM_T004GT', 'ERNAM_T005', 'ERNAM_T005GT', 'ERNAM_T006', 'ERNAM_T006GT', 'ERNAM_T007', 'ERNAM_T007GT', 'ERNAM_T008', 'ERNAM_T008GT', 'ERNAM_T010', 'ERNAM_T010GT', 'ERNAM_T011', 'ERNAM_T011GT', 'ERNAM_T012', 'ERNAM_T012GT', 'ERNAM_T013', 'ERNAM_T013GT', 'ERNAM_T014', 'ERNAM_T014GT', 'ERNAM_T015', 'ERNAM_T015GT', 'ERNAM_T016', 'ERNAM_T016GT', 'ERNAM_T017', 'ERNAM_T017GT', 'ERNAM_T018', 'ERNAM_T018GT', 'ERNAM_T019', 'ERNAM_T019GT', 'ERNAM_T020', 'ERNAM_T020GT', 'ERNAM_T021', 'ERNAM_T021GT', 'ERNAM_T022', 'ERNAM_T022GT', 'ERNAM_T023', 'ERNAM_T023GT', 'ERNAM_T024', 'ERNAM_T024GT', 'ERNAM_T025', 'ERNAM_T025GT', 'ERNAM_T026', 'ERNAM_T027', 'ERNAM_T027GT', 'ERNAM_T028', 'ERNAM_T028GT', 'ERNAM_T029', 'ERNAM_T029GT', 'ERNAM_T030', 'ERNAM_T030GT', 'ERNAM_T031', 'ERNAM_T031GT', 'ERNAM_T032', 'ERNAM_T032GT', 'ERNAM_T033', 'ERNAM_T033GT', 'ERNAM_T034', 'ERNAM_T034GT', 'ERNAM_T035', 'ERNAM_T036', 'ERNAM_T800', 'ERNAM_T801', 'ERNAM_T801GT', 'ERNAM_T802', 'ERNAM_T802GT', 'ERNAM_T803', 'ERNAM_T803GT', 'ERNAM_T804', 'ERNAM_T804GT', 'ERNAM_T805', 'ERNAM_T805GT', 'ERNAM_T806', 'ERNAM_VSANTANA', 'ERNAM_YRODRIGUEZ'],
    'date-cols': ['MBDAT','AEDAT','ERDAT','WADAT','LDDAT','TDDAT','LFDAT','KODAT','BLDAT','WADAT_IST', 'FKDAT'],
    'empty-cols':['MATWA', 'CHARG', 'LICHN', 'KDMAT', 'PRODH', 'GEWEI', 'VOLEH', 'UEBTK', 'CHSPL', 'FAKSP', 'LGPBE', 'VBELV', 'VBTYV', 'VGSYS', 'UPFLU', 'LGNUM', 'LISPL', 'LGTYP', 'LGPLA', 'BWTEX', 'KZDLG', 'BDART', 'PLART', 'XCHPF', 'XCHAR', 'VGREF', 'POSAR', 'BWTAR', 'EANNR', 'GSBER', 'VKBUR', 'VKGRP', 'FMENG', 'STAFO', 'SOBKZ', 'BZIRK', 'AUTLF', 'ABLAD', 'INCO1', 'INCO2', 'EXPKZ', 'ROUTE', 'FAKSK', 'LIFSK', 'KNFAK', 'TPQUA', 'TPGRP', 'KUNAG', 'KDGRP', 'BEROT', 'GRULG', 'LSTEL', 'PERFK', 'ROUTA', 'KNUMV', 'VERUR', 'COMMN', 'EXNUM', 'VKOIV', 'VTWIV', 'SPAIV', 'FKAIV', 'PIOIV', 'KUNIV', 'KKBER', 'KNKLI', 'GRUPP', 'SBGRP', 'CTLPC', 'CMWAE', 'BOLNR', 'LIFNR', 'TRATY', 'TRAID', 'XABLN', 'TRSPG', 'TPSID', 'LIFEX', 'TERNR', 'KALSM_CH', 'KLIEF', 'KALSP', 'KNUMP', 'AULWE'],
    'zero-cols':['NTGEW', 'BRGEW', 'VOLUM', 'UEBTO', 'UNTTO', 'POSNV', 'UEPOS', 'BWLVS',
       'GRKOR', 'ANTLF', 'VBEAF', 'VBEAV', 'WAVWR', 'KZWI1', 'KZWI2', 'KZWI3',
       'KZWI4', 'KZWI5', 'KZWI6', 'LPRIO', 'STZKL', 'STZZU', 'BTGEW', 'VBEAK',
       'AMTBL', 'NETWR'],
    'zero-date-cols': ['WADAT_IST'],
    'month-cols': ['MBDAT_MONTH','AEDAT_MONTH','ERDAT_MONTH','WADAT_MONTH','LDDAT_MONTH','TDDAT_MONTH','LFDAT_MONTH','KODAT_MONTH','BLDAT_MONTH','WADAT_IST_MONTH', 'FKDAT_MONTH'],
    'target-cols': ['VSTEL']
}

###################################################################
###################################################################
########################### AUX CLASSES ###########################
###################################################################
###################################################################

class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

class AcopioModel(nn.Module):
    def __init__(self, num_classes=42,  start_features = 267):
        super().__init__()

        self.num_classes = num_classes
        self.nns = 64
        self.start_features = start_features
        self.layer = nn.Sequential(
            nn.Linear(self.start_features, self.nns),
            nn.ReLU(),
            nn.Linear(self.nns, self.nns),
            nn.ReLU(),
            nn.Linear(self.nns, self.num_classes)
        )
    def forward(self, x):
        return self.layer(x)

class MyDataset(Dataset):
    def __init__(self,data_dir, split='train', batch_size = 1 ):

        super().__init__()
        self.split = split
        self.inputs = []
        self.labels = [] 
        self.bs = batch_size
        # LOAD SPLIT CSV FILE
        
        #self.root_dir = '/content/drive/My Drive/AcopioCockpit/Dataset/ds/dataset-'
        self.root_dir = data_dir
        #self.input_dir= self.root_dir + '/data-' + self.split +'.csv'
        #self.label_dir= self.root_dir + '/data-' + self.split +'.csv'
        
        input_files = [ os.path.join(data_dir, file) for file in os.listdir(data_dir) ]
        
        if len(input_files) == 0:
            raise ValueError('No se encuentra el archivo especificado en el canal {}'.format(self.split))
        
        with open(input_files[0]) as file_input:
            csv_file_input = csv.reader(file_input, delimiter=',')
            for row in csv_file_input:
                
                self.labels.append(row[0])
                input = list(map(float, row[1:]))
                self.inputs.append(input)
                """
                for idx, value  in enumerate(row.items()):
                    if idx == 0:
                        labels_added = labels_added +1
                        self.labels.append(value)
                    else:
                        input.append(value)
                self.inputs.append(input)
                print('input: {}'.format(input))
                if labels_added > 1:
                    print(labels_added)
                """
        #print(np.unique(self.labels))
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        
        input = self.inputs[index]
        label = self.labels[index]
        # Parseamos el input en una lista de floats
        input = list(map(float, input))
        #label = list(map(float, label))
        label= [float(label)]
        return torch.tensor(input), torch.tensor(label)

###################################################################
###################################################################
####################### INTERFACE FUNCTIONS #######################
###################################################################
###################################################################    

# Defines the model that will be trained
def model_fn(model_dir):
        
    model = AcopioModel()
    if model_dir is None:
        model_dir = os.environ['SM_MODEL_DIR']
    print('Leemos modelo de {}'.format(model_dir))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    print('Modelo cargado')
    return model


# Takes request data and deserializes the data into an object for prediction.
def input_fn(request_body, request_content_type): 
    
    print('Inicio input_fn')
    print(request_content_type)
    if request_content_type == 'application/python-pickle':
        return torch.load(BytesIO(request_body))
    elif request_content_type == 'application/json':
        return parse_prediction_request(request_body)
    elif request_content_type == 'text/csv':
        return parse_prediction_request(request_body)
    # return None


# Takes the deserialized request object and performs inference against the loaded model.
def predict_fn(input_object, model):
    
    print('Inicio predict_fn')
    print(input_object)
    if model is None:
        model = model_fn()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(input_object)
        print(prediction)
    return prediction

# Takes the result of prediction and serializes this according to the response content type.
def output_fn(prediction, response_content_type):
    
    print('Inicio output_fn')
    print(prediction)
    return prediction

def get_random_date():
    start_date = datetime.date(2010, 1, 1)
    end_date = datetime.date(2020, 8, 25)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date

def get_zero_date_cols (df):
    return [col for col in df.columns if df[col].dtype == 'object' and df[col].str.contains(r'^0000-00-00$').any()]


def parse_prediction_request(body):
    
    print('Inicio parse prediction request')
    print(body)
    
    print('Leemos el body')

    data = StringIO(body)
    df = pd.read_csv(data, sep=",", header=0)

    print('Imprimimos body')
    print(df)

    # Eliminamos las columnas con datos unívocos
    df = df.drop(vars['remove-cols'], axis=1)
    #print(len(df.columns))

    print('Empty cols')
    df = df.drop(vars['empty-cols'], axis=1)
    #print('empty cols: {}, final num: {}'.format(len(vars['empty-cols']),len(df.columns)))

    #Eliminamos columnas vacias
    print('Zero cols')
    df = df.drop(vars['zero-cols'],axis=1)
    #print('zero cols: {}, final num: {}'.format(len(vars['zero-cols']),len(df.columns)))

    #Borramos el target
    print('Borrar target')
    df = df.drop('VSTEL', axis=1)
    print('borrar target cols: {}, final num: {}'.format(1, len(df.columns)))

    # Nos quedamos unicamente con el mes de las fechas creando otra columna y eliminamoslas originales
    print('Transformamos fechas en meses')
    for col in vars['date-cols']:
        month = pd.DatetimeIndex(df[col]).month
        df[col+"_MONTH"] = month
        df = df.drop(col,axis=1)
    print('month cols: {}, final num: {}'.format(len(vars['date-cols']), len(df.columns)))


    print('Codificamos one-hot')
    #df_onehot = pd.get_dummies(df, columns=vars['onehot-cols'], drop_first=True)
    df_onehot = pd.DataFrame(columns=vars['onehot-vals'],dtype='int64')
    df_onehot = df_onehot.append(pd.Series(0, index=df_onehot.columns), ignore_index=True)

    # Para las one.hot, buscamos la columna nueva y ponemos el 1 si existe
    for col in vars['onehot-cols']:
        col_value = df[col][0] # Si entra un batch habria que cambiar el algoritmo 
        colname = col+"_"+str(col_value)
        if colname in df_onehot.columns:
            df_onehot.at[0,colname]=50


    # Borramos las one-hot despues de setear el 1 en su correspondiente
    df = df.drop(vars['onehot-cols'], axis=1)
    print('onehot cols: {}, final num: {}'.format(len(vars['onehot-cols']),len(df_onehot.columns)))
    print('Construimos df')
    df_trans = pd.concat([df[vars['month-cols']], df_onehot], axis=1, sort=False)

    print('Final num cols: {}'.format(len(df_trans.columns)))
    print(df_trans)
    #data = df_trans.to_json()
    print('Convertimos en array')
    data = [x for x in df_trans.to_numpy()]
    #print(data)
    print('Salimos')
    data = torch.from_numpy(df_trans.to_numpy())
    data = torch.squeeze(data)
    print(data)
    return data;


###################################################################
###################################################################
############################ TRAINING #############################
###################################################################
###################################################################


    
def train_one_epoch(train_loader, net, optimizer, criterion, hparams):

  # Activate the train=True flag inside the model
    net.train()
  
    device = hparams['device']
    batch_size = hparams['batch_size']
    train_loss, train_accs = 0, 0
    train_f1s = {}
    times_per_step_iteration = []
    times_per_metric_iteration = []
    times_per_iteration = []
    for batch_index, (input, target) in enumerate(train_loader):
        #Arrancamos temporizador
        #start_total.record()
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        
        #Arrancamos temporizador
        #start.record()
        output = net(input)
        
        target = target.long()
        target = torch.squeeze(target,1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
         
        train_accuracy = calculate_accuracy(output, target) #, predicted
        #print('Accuracy: {}'.format(train_accuracy))
        train_accs  += train_accuracy
        
        train_f1 = calculate_f1(output, target) #, predicted
        
        for key in train_f1:
            if key not in train_f1s:
                train_f1s[key] = train_f1[key]
            else:
                train_f1s[key] = train_f1s[key] + train_f1[key]
       
        #print(train_f1s)
        
    train_loss = train_loss / (len(train_loader.dataset) / batch_size)
    train_accs = 100 * (train_accs / (len(train_loader.dataset) / batch_size))
    train_f1s = convert_batched_f1(train_f1s, (len(train_loader.dataset) / batch_size))
  
    mF1 = get_mf1(train_f1s)
    #print(mF1)
    mF1_desc = f1_to_string(train_f1s, hparams)
    return train_loss, train_accs, mF1, mF1_desc
      
    
def val_one_epoch(val_loader, net,criterion, params):

    net.eval()
    device = params['device']
    batch_size = params['batch_size']
    val_loss, val_accs = 0, 0
    val_f1s = {}
    pred = 0
    with torch.no_grad():
        for batch_index, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            target = target.long()
            target = torch.squeeze(target,1)
            loss = criterion(output, target)
            val_loss += loss

            # compute number of correct predictions in the batch
            val_accuracy = calculate_accuracy(output, target)
            val_accs += val_accuracy
            val_f1 = calculate_f1(output, target)
            
            for key in val_f1:
                if key not in val_f1s:
                    val_f1s[key] = val_f1[key]
                else:
                    val_f1s[key] = val_f1s[key] + val_f1[key] 
                    
    # Average acc across all correct predictions batches now
    val_loss = val_loss / (len(val_loader.dataset) / batch_size)
    val_accs = 100 * (val_accs / (len(val_loader.dataset) / batch_size))
    val_f1s = convert_batched_f1(val_f1s, (len(val_loader.dataset) / batch_size))
  
    mF1 = get_mf1(val_f1s)
    #print(mF1)
    mF1_desc = f1_to_string(val_f1s, params)
    return val_loss, val_accs, mF1, mF1_desc    
    
    
###################################################################
###################################################################
############################ METRICS ##############################
###################################################################
###################################################################
    
def calculate_accuracy(output, target):

      # Sumar el numero de muestras que ha clasificado bien
    _, predicted = torch.max(output.data, 1)
    correctos_batch, fallidos_batch = predicted.eq(target.data).sum().item(), predicted.ne(target.data).sum().item()
    result = correctos_batch/ (correctos_batch + fallidos_batch)
    return result


def calculate_f1(output, target):

    output, target = output.cpu(), target.cpu()
    _, prediction = torch.max(output, 1)
    num_tiendas = np.unique(np.concatenate((prediction, target), axis = 0))
    num_tiendas = np.sort(num_tiendas)
    class_f1 = {}
    # print('prediction: {}'.format(prediction))
    for tienda_num in num_tiendas:
        
        tienda = torch.Tensor([tienda_num])
        tienda = tienda.long()
        torch.eq(target, tienda)
        torch.eq(prediction, tienda)
        tp = np.logical_and(torch.eq(prediction, tienda), torch.eq(target, tienda)).long().sum().item() # Cuantas targets son la tienda y nuestras predicciones son la tienda     
        fn = np.logical_and(torch.ne(prediction, tienda), torch.eq(target, tienda)).long().sum().item() # Cuantos targets son la tienda y nosotros hemos fallado
        fp = np.logical_and(torch.eq(prediction, tienda), torch.ne(target, tienda)).long().sum().item() # Cuantos targets no son la tienda y nosotros hemos dicho que sí
        
        f1, precision, recall = 0, 0, 0
        # Calculamos
        if tp + fp != 0:
            precision = tp / (tp + fp)

        if tp + fn != 0:
            recall = tp / (tp + fn)

        if tp + fn + fp != 0:
            f1 = tp / (tp + (fp + fn)/2)
            class_f1[tienda_num] = [f1]
    
    return class_f1

def convert_batched_f1(input_f1, length):

    for key in input_f1:
        key_len = len(input_f1[key])
        suma = sum(input_f1[key])
        input_f1[key] = suma / key_len

    return input_f1

def get_mf1(input_f1):

    count = 0
    _sum = 0
    for key in input_f1:
        count += 1
        _sum += input_f1[key]

    return _sum/count


def f1_to_string(f1_classes, params):

    #print(f1_classes)
    sorted_f1 = {k: v for k, v in sorted(f1_classes.items(), key=lambda item: item[1], reverse=True)}
    string = ['F1 per class\n']
    #string.append('---------------------\n')
    string.append("Classes\t Values\n")
    string.append('-------   ------\n')
    bprinted_threshold= False
    tiendas_f1 = {}
    for key, value in sorted_f1.items():
        for tienda, valor in params['class_mapping'].items():
            if key == valor:
                tiendas_f1[tienda] = value

    for key, value in tiendas_f1.items():
        string.append(str(key) + '\t' + str(' %.2f' % value) + '\n')
  
    return ''.join(string)


def save_model(model, path):
    torch.save(model.state_dict(), path+"/model.pth")# model.state_dict()

def load_model(path):
    
    return torch.load(path+"/model.pth")


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:

    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft  = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


###################################################################
###################################################################
############################## MAIN ###############################
###################################################################
###################################################################


def main (train_data_dir, val_data_dir, model_dir, epochs = 10, batch_size=25, lr = 0.05):


    params = {
        'best_model_path': model_dir,
        'device': torch.device("cuda"),
        'batch_size': batch_size,
        'focal_loss_alpha':0.25,
        'focal_loss_gamma':2.0,
        'adam_learning_rate': 1E-3,
        'adam_aux_learning_rate': 5E-4,
        'adam_weight_decay': 1E-4,
        'num_epochs': epochs,
        'class_mapping': {'T001': 0, 'T002': 1, 'T004': 2, 'T005': 3, 'T006': 4, 'T007': 5, 'T008': 6, 'T010': 7, 'T011': 8, 'T012': 9, 'T013': 10, 'T014': 11, 'T015': 12, 'T016': 13, 'T017': 14, 'T018': 15, 'T019': 16, 'T020': 17, 'T021': 18, 'T022': 19, 'T023': 20, 'T024': 21, 'T025': 22, 'T026': 23, 'T027': 24, 'T028': 25, 'T029': 26, 'T030': 27, 'T031': 28, 'T032': 29, 'T033': 30, 'T034': 31, 'T035': 32, 'T036': 33, 'T300': 34, 'T800': 35, 'T801': 36, 'T802': 37, 'T803': 38, 'T804': 39, 'T805': 40, 'T806': 41}
    }
    params['device'] = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    
    #Definimos temporizadores
    #Arrancamos el temporizador
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)
    """

    # Creamos los datasets y dataloaders
    train_dataset = MyDataset(split='train', data_dir=train_data_dir)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers = 4)

    val_dataset = MyDataset(split='val', data_dir=val_data_dir)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers = 4)

    #test_dataset = MyDataset(split='test')
    #test_loader = utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers = 4)
    

    network = AcopioModel()

    network.to(params['device'])
    net_params = network.parameters()

    #summary(network, input_size=(1, 267))
    #criterion = torch.nn.CrossEntropyLoss()

    criterion = FocalLoss(gamma=params['focal_loss_gamma'], alpha=params['focal_loss_alpha'], reduction='mean').cuda()
    optimizer = torch.optim.RMSprop(net_params, lr=params['adam_learning_rate'], eps=1e-08, weight_decay=params['adam_weight_decay'])

    print('Dataset train samples: {}, dataset val samples: {}'.format(len(train_loader.dataset), len(val_loader.dataset)))

    best_epoch_mf1 = 0
    train_losses, train_acc_hist, train_mf1s = [], [], []
    val_losses, val_acc_hist, val_mf1s = [], [], []
    
    for epoch in range(1, params['num_epochs'] +1):

        # Compute & save the average training loss for the current epoch
        print('#################### Epoch: {} ####################\n'.format(epoch))

        print('Inicio training epoch {}'.format(epoch))
        train_loss, train_acc, train_mf1, train_f1_desc = train_one_epoch(train_loader, network, optimizer, criterion, params)  
        print('Training set: Average loss {:.4f}, Mean accuracy: {:.2f}%, mF1: {:.4f}\n'.format(train_loss, train_acc, train_mf1)) # train_f1_desc

        train_losses.append(train_loss)
        train_mf1s.append(train_mf1)
        train_acc_hist.append(train_acc)

        print('Inicio validacion epoch {}'.format(epoch))                
        val_loss, val_acc, val_mf1, val_f1_desc = val_one_epoch(val_loader, network,criterion, params)
        print('Validation set: Average loss: {:.4f}, Mean accuracy: {:.2f}%, mF1: {:.4f}\n{}\n'.format(val_loss, val_acc, val_mf1, val_f1_desc))
        
        if val_mf1 > best_epoch_mf1:
            best_epoch_mf1 = val_mf1
            print('Guardamos el modelo en epoch {} ( mIoU {:.2f})'.format(epoch, val_mf1))
            save_model(network, params['best_model_path'])
  
        val_losses.append(val_loss)
        val_mf1s.append(val_mf1)
        val_acc_hist.append(val_acc)
    print('-----------------------------------------\n')        
    print('Fin del entrenamiento\n')
        
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    
    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()
    
    print('Iniciamos')
    
    input_files = [ os.path.join(args.train_dir, file) for file in os.listdir(args.train_dir) ]
    print('Leemos datos de {}'.format(input_files[0]))
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train_dir, "train"))
    
    
    count= 0
    with open(input_files[0]) as file_input:
            csv_file_input = DictReader(file_input, delimiter=',')
            for row in csv_file_input:
                input = []
                for key, value in row.items(): 
                    count = count + 1
    print('Terminamos de leer {} lineas'.format(count))
    
    main(train_data_dir = args.train_dir, val_data_dir = args.val_dir, epochs = args.epochs, batch_size= args.batch_size, lr = args.learning_rate, model_dir= args.model_dir)
    
    