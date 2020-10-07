# -*- coding: utf-8 -*-
"""
Created on 2020
@author: Serna
"""
#--------------------- LIBRERÍAS ----------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
import matplotlib.ticker as plticker
import pylab 
import scipy.stats as stats
from scipy.interpolate import Rbf
import matplotlib
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.special import boxcox, inv_boxcox
from matplotlib import rc
from itertools import repeat
import math
matplotlib.use('Agg')
sns.set(style="white")
import random
random.seed(9)
from sklearn import preprocessing
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors


# ------------- CREACION DEL DATAFRAME -------------- #
d = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Estaciones/Historico.csv', dtype = {'station': str, 'notes': str, 'timestamp': str})
d.timestamp = pd.to_datetime(d.timestamp, format='%d-%b-%y %H')

d2 = d.copy()
d2 = d2.set_index(d.timestamp)
d_all = d2.drop(["timestamp","station", "valid", "notes"], axis=1)

# ----------------- BARPLOT VARIABLE ----------------#
list_nam = []
list_val = []
list_nan = []

for i in d_all.columns:
    nan = d_all[i].isnull().sum().sum()
    val = len(d_all) - nan
    
    list_nam.append(i)
    list_val.append(val)
    list_nan.append(nan)

"""r = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
data = {'greenBars': list_val, 'orangeBars': list_nan}
df = pd.DataFrame(data)
 
totals = [i+j for i,j in zip(df['greenBars'], df['orangeBars'])]
greenBars = [i / j * 100 for i,j in zip(df['greenBars'], totals)]
orangeBars = [i / j * 100 for i,j in zip(df['orangeBars'], totals)]
barWidth = 0.85


plt.bar(r, greenBars, color='b', edgecolor='white', width=barWidth, label=" % datos")
plt.bar(r, orangeBars, bottom=greenBars, color='r', edgecolor='white', width=barWidth, label="% nan")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
plt.xticks(r, list_nam, rotation=90)
plt.xlabel("Variables")
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()]) 
plt.tight_layout()
plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/BarPlot_All.png', format='png', dpi=1000)
plt.show()
plt.close()

 
barWidth = 0.35
r1 = np.arange(len(list_val))
r2 = [x + barWidth for x in r1]
plt.bar(r1, list_val, color='b', width=barWidth, edgecolor='white', label="Datos")
plt.bar(r2, list_nan, color='r', width=barWidth, edgecolor='white', label="Nan")
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
plt.xticks(r, list_nam, rotation=90)
plt.xlabel("Variables")
plt.tight_layout()
plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/BarPlot_All_2.png', format='png', dpi=1000)
plt.show()
plt.close()


group_names = list_nam
group_size = list(repeat(len(d_all), 15))

subgroup_size = []
for i in range(0,15):
    subgroup_size.append(list_val[i])
    subgroup_size.append(list_nan[i])
        
colores = ['skyblue','skyblue','skyblue','skyblue','skyblue','skyblue','skyblue','skyblue','skyblue','skyblue',
           'skyblue','skyblue','skyblue','skyblue','skyblue',]
colores_2 = ['b','r','b','r','b','r','b','r','b','r','b','r','b','r','b','r','b','r','b','r',
             'b','r','b','r','b','r','b','r','b','r',]
explodes = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
labels= ['Datos','Nan']

fig, ax = plt.subplots()
mi_pie, _ = ax.pie(group_size, radius=1.4, labels=group_names, colors=colores, explode=explodes)
plt.setp( mi_pie, width=0.3, edgecolor='white')
mi_pie2, _ = ax.pie(subgroup_size, radius=1.4-0.3, labeldistance=0.7, colors=colores_2, textprops={'fontsize': 18})
plt.setp( mi_pie2, width=0.4, edgecolor='white')
plt.margins(0,0)
plt.legend(mi_pie2, labels, loc='center')
plt.tight_layout()
plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/BarPie.png', format='png', dpi=1000)
plt.show()
plt.close()

labels = 'Datos', 'Nan'
valores = sum(list_val)
nans = sum(list_nan)
tamaño = [valores, nans]

fig, ax = plt.subplots()
ax.pie(tamaño, labels=labels, autopct='%1.1f%%', colors=['b','r'])
plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/BarPie_2.png', format='png', dpi=1000)
plt.show()
plt.close()"""

# ---------------- DATA VARIABLE ------------------- #
d_all_2 = d2.drop(["valid", "notes"], axis=1)

filtro_variable=[]
for i in d_all_2.columns:
    var = d_all_2.pivot(columns='station', values=i)
    filtro_variable.append(var)

filtro_variable = filtro_variable[2:]
        # --------- HISTOGRAMA POR ESTACIÓN --------- #
columnas_names= d_all_2.columns

sns.set(font_scale=1)
for i in range(0,15):
    j=i+2
    histograma = fil_var_2[i].hist(bins=20, figsize=(11,11))
    nombre_estacion = columnas_names[j]
    #plt.suptitle(nombre_estacion, size=16)
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/Hist/histogram_'+'_'+str(j)+'_'+str(nombre_estacion)+'_3.png', format='png', dpi=1000)
    plt.show()
    plt.close()

        # ----- HISTOGRAMA POR ESTACIÓN JUNTAS ----- #
sns.set(font_scale=0.7)
for i in range(len(fil_var_2)):
    j=i+2
    ax = fil_var_2[i].plot.hist(bins=20, alpha=0.6)
    plt.ylabel(' ')
    ax.legend(title="Estaciones", loc = 'rigth')
    nombre_estacion = columnas_names[j]
    nombre_estacion_2 = names_var[i]
    plt.xlabel(nombre_estacion_2, size=12)
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/Hist/histogram_'+'_'+str(j)+'_'+str(nombre_estacion)+'_2'+'.png', format='png', dpi=1000)
    plt.show()
    plt.close()


# ------------- TRATAMIENTO DE NAN's --------------- #
d_aux = d_all_2.iloc[list(np.where(d2["timestamp"] == '2016-01-01 00:00:00')[0])[0]:]
d_aux = d_aux.drop(["timestamp"], axis=1)

fil_var=[]
for i in d_aux.columns:
    var = d_aux[i]
    var = d_aux.pivot(columns='station', values=i)
    fil_var.append(var)    
fil_var = fil_var[1:16]

fil_var_2 = fil_var.copy()
for i in range(0,15):
    fil_var_2[i]=fil_var_2[i].interpolate(method='time')
    
    m = fil_var_2[i].mean(axis=1)
    for j, col in enumerate(fil_var_2[i]):
        fil_var_2[i].iloc[:, j] = fil_var_2[i].iloc[:, j].fillna(m)

# --- SE ELIMINA (pressure, rainfall, humidity, solar, temperature) --- #
"""fil_var_2=[fil_var_2[0], fil_var_2[1], fil_var_2[2], fil_var_2[3], fil_var_2[4], fil_var_2[5],
           fil_var_2[6], fil_var_2[10], fil_var_2[13], fil_var_2[14]]"""

fil_var_3=[]
for i in d_aux.columns:
    var = d_aux[i]
    var = d_aux.pivot(columns='station', values=i)
    fil_var_3.append(var)
    
fil_var_3 = fil_var_3[1:16]
  # ----------- TIME SERIES 16 --------------------#
"""for i in range (0,15):
    plt.style.use('seaborn')
    axes = fil_var_2[i].plot(marker='.', markersize =1, linewidth=0.5,
                          linestyle='-', figsize=(9,15), subplots=True)
    for ax in axes:
        start, end = ax.get_ylim()
        loc = plticker.MultipleLocator(base=400) 
        ax.yaxis.set_major_locator(loc)
    nombre = list_nam[i]
    plt.tight_layout()
    plt.xlabel(' ')
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/Serie/Serie_'+str(nombre)+'.png', format='png', dpi=1000)
    plt.show()
    

for i in range (0,15):
    plt.style.use('seaborn')
    axes = fil_var_3[i].plot(marker='.', markersize =1, linewidth=0.5,
                          linestyle='-', figsize=(9,15), subplots=True)
    for ax in axes:
        start, end = ax.get_ylim()
        loc = plticker.MultipleLocator(base=400) 
        ax.yaxis.set_major_locator(loc)
    nombre = list_nam[i]
    plt.tight_layout()
    plt.xlabel(' ')
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/Serie/Serie_Nans_'+str(nombre)+'.png', format='png', dpi=1000)
    plt.show()"""
  # -------- CORRELATION ALL x ALL ------------------#
corr_aux1 = []
for v in range(len(fil_var_2)):
    corr_aux2 =[]
    for w in range(len(fil_var_2)):
        m = fil_var_2[v]*fil_var_2[w]
    
        #------matriz de correlacion ---#
        corr = m.corr(method='pearson')
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        corr_aux2.append(corr)
    corr_aux1.append(corr_aux2)

names_var =['CO','NO','NO2','NOX','O3','PM10','PM2.5','Presión','Precipitación','Humedad','SO2',
            'Radiación','Temperatura','Velocidad','Dirección']
for i in range(len(fil_var_2)):
    sns.set(font_scale=0.4)
    fig, ax = plt.subplots(4,4, figsize=(5,4))
    sns.heatmap(corr_aux1[i][0], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[0,0])#.set_title(str(names_var[i]+' contra CO'))
    ax[0,0].set_ylabel('')    
    ax[0,0].set_xlabel(str(names_var[i]+' contra CO'))
    sns.heatmap(corr_aux1[i][1], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[0,1])
    ax[0,1].set_ylabel('')    
    ax[0,1].set_xlabel(str(names_var[i]+' contra NO'))
    sns.heatmap(corr_aux1[i][2], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[0,2])
    ax[0,2].set_ylabel('')    
    ax[0,2].set_xlabel(str(names_var[i]+' contra NO2'))
    sns.heatmap(corr_aux1[i][3], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[0,3])
    ax[0,3].set_ylabel('')    
    ax[0,3].set_xlabel(str(names_var[i]+' contra NOX'))
    sns.heatmap(corr_aux1[i][4], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[1,0])
    ax[1,0].set_ylabel('')    
    ax[1,0].set_xlabel(str(names_var[i]+' contra O3'))
    sns.heatmap(corr_aux1[i][5], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[1,1])
    ax[1,1].set_ylabel('')    
    ax[1,1].set_xlabel(str(names_var[i]+' contra PM10'))
    sns.heatmap(corr_aux1[i][6], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[1,2])
    ax[1,2].set_ylabel('')    
    ax[1,2].set_xlabel(str(names_var[i]+' contra PM2.5'))
    sns.heatmap(corr_aux1[i][7], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[1,3])
    ax[1,3].set_ylabel('')    
    ax[1,3].set_xlabel(str(names_var[i]+' contra Presión'))
    sns.heatmap(corr_aux1[i][8], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[2,0])
    ax[2,0].set_ylabel('')    
    ax[2,0].set_xlabel(str(names_var[i]+' contra Precipitación'))
    sns.heatmap(corr_aux1[i][9], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[2,1])
    ax[2,1].set_ylabel('')    
    ax[2,1].set_xlabel(str(names_var[i]+' contra Humedad'))
    sns.heatmap(corr_aux1[i][10], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[2,2])
    ax[2,2].set_ylabel('')    
    ax[2,2].set_xlabel(str(names_var[i]+' contra SO2'))
    sns.heatmap(corr_aux1[i][11], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[2,3])
    ax[2,3].set_ylabel('')    
    ax[2,3].set_xlabel(str(names_var[i]+' contra Radiación'))
    sns.heatmap(corr_aux1[i][12], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[3,0])
    ax[3,0].set_ylabel('')    
    ax[3,0].set_xlabel(str(names_var[i]+' contra Temperatura'))
    sns.heatmap(corr_aux1[i][13], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[3,1])
    ax[3,1].set_ylabel('')    
    ax[3,1].set_xlabel(str(names_var[i]+' contra Velocidad'))
    sns.heatmap(corr_aux1[i][14], mask=mask, cmap='seismic', center=0, vmin=-1, vmax=1, square=True, linewidths=.1, cbar_kws={"shrink": .5},annot=False, xticklabels=False, yticklabels=False,
                ax=ax[3,2])
    ax[3,2].set_ylabel('')    
    ax[3,2].set_xlabel(str(names_var[i]+' contra Dirección'))
    ax[-1, -1].axis('off')
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Tesis/Correlacion/Corr_'+str(list_nam[i])+'_all_final.png', format='png', dpi=1000)
    plt.show()
    plt.close()


# --DATAS FRAME DE ESTACIONES RANDOM E INTERPOLADAS-- #
estaciones_selec = 12  # - estaciones - #
seleccionados = []
interpolados = []

for i in range(len(fil_var_3[0])):
    renglones = np.arange(13)
    np.random.shuffle(renglones)
    
    seleccionar = renglones[0:estaciones_selec]
    seleccionar  = list(seleccionar)
    seleccionados.append(seleccionar)
    
    interpolar = renglones[estaciones_selec:13]
    interpolar = list(interpolar)
    interpolados.append(interpolar)
    
data_seleccionados = pd.DataFrame(seleccionados)
data_interpolados = pd.DataFrame(interpolados)

# ----- DATA VALORES CON POSICIONES RANDOM ----- #
for i in range(0,10):
    fil_var_2[i] = fil_var_2[i].reset_index(drop=True)   
    
for i in range(0,10):
     fil_var_2[i] = fil_var_2[i].rename(columns={"Centro":0 ,"Noreste":1, "Noreste2":2, "Noroeste":3,
              "Noroeste2":4, "Norte":5, "Norte2":6, "Sur":7, "Sureste":8, "Sureste2":9, "Sureste3":10, 
             "Suroeste":11, "Suroeste2":12})

total_lista_valores = []
for j in range(0,10):
    lista_valores = []
    for i in range(len(fil_var_3[0])):
        lista_aux = []
        for k in range(len(seleccionar)):
            indice_estacion = data_seleccionados[k][i]
            valor_estacion = fil_var_2[j][indice_estacion][i]
            lista_aux.append(valor_estacion)
    
        lista_valores.append(lista_aux)
    data_valores = pd.DataFrame(lista_valores)
    total_lista_valores.append(data_valores)
    
# ------- MIN-MAX DE DATOS INTERPOLADOS --------- #
for s in range(0, 10):
    for i in range (0, len(seleccionados[0])):
        print(s)
        for j in range(0, len(total_lista_valores[0][0])):
            if s == 0:
                if total_lista_valores[s][i][j] <= 0.05:
                    total_lista_valores[s][i][j] = 0.05
            elif s == 1:
                if total_lista_valores[s][i][j] <= 1:
                    total_lista_valores[s][i][j] = 1
            elif s == 2:
                if total_lista_valores[s][i][j] <= 1:
                    total_lista_valores[s][i][j] = 1
            elif s == 3:
                if total_lista_valores[s][i][j] <= 1:
                    total_lista_valores[s][i][j] = 1
            elif s == 4:
                if total_lista_valores[s][i][j] <= 1:
                    total_lista_valores[s][i][j] = 1
            elif s == 5:
                if total_lista_valores[s][i][j] <= 2:
                    total_lista_valores[s][i][j] = 2
            elif s == 6:
                if total_lista_valores[s][i][j] <= 2:
                    total_lista_valores[s][i][j] = 2
            elif s == 7:
                if total_lista_valores[s][i][j] <= 1:
                    total_lista_valores[s][i][j] = 1
            elif s == 8:
                if total_lista_valores[s][i][j] <= 0:
                    total_lista_valores[s][i][j] = 0.01
            elif s == 9:
                if total_lista_valores[s][i][j] <= 0.0:
                    total_lista_valores[s][i][j] = 0.01
                    
# ------------ NORMALIZADO DE LOS DATOS ------------- #
list_m_pearson = []
for s in range(0, 10):
    m_pearson = stats.boxcox_normmax(total_lista_valores[s].values.flatten(), method='pearsonr')
    total_lista_valores[s] = pd.DataFrame(stats.boxcox(total_lista_valores[s], m_pearson))
    list_m_pearson.append(m_pearson)
                        
#- DATA REAL PARA COMPARAR CONTRA INTERPOLADOS -#
total_lista_estimados = []
for k in range(0,10):
    lista_estimados = []
    for i in range(len(data_valores)):
        lista_aux_2 = []
        for j in range(len(interpolar)):
            indice_interpolar = data_interpolados[j][i]
            valor_interpolar = fil_var_2[k][indice_interpolar][i]
            lista_aux_2.append(valor_interpolar)
        lista_estimados.append(lista_aux_2)
    
    data_real = pd.DataFrame(lista_estimados)
    total_lista_estimados.append(data_real)
    
# ------- MIN-MAX DE DATOS REALES --------- #
for s in range(0, 10):
    for i in range (0, len(interpolados[0])):
        print(s)
        for j in range(0, len(total_lista_estimados[0][0])):
            if s == 0:
                if total_lista_estimados[s][i][j] <= 0.05:
                    total_lista_estimados[s][i][j] = 0.05
            elif s == 1:
                if total_lista_estimados[s][i][j] <= 1:
                    total_lista_estimados[s][i][j] = 1
            elif s == 2:
                if total_lista_estimados[s][i][j] <= 1:
                    total_lista_estimados[s][i][j] = 1
            elif s == 3:
                if total_lista_estimados[s][i][j] <= 1:
                    total_lista_estimados[s][i][j] = 1
            elif s == 4:
                if total_lista_estimados[s][i][j] <= 1:
                    total_lista_estimados[s][i][j] = 1
            elif s == 5:
                if total_lista_estimados[s][i][j] <= 2:
                    total_lista_estimados[s][i][j] = 2
            elif s == 6:
                if total_lista_estimados[s][i][j] <= 2:
                    total_lista_estimados[s][i][j] = 2
            elif s == 7:
                if total_lista_estimados[s][i][j] <= 1:
                    total_lista_estimados[s][i][j] = 1
            elif s == 8:
                if total_lista_estimados[s][i][j] <= 0:
                    total_lista_estimados[s][i][j] = 0.01
            elif s == 9:
                if total_lista_estimados[s][i][j] <= 0.0:
                    total_lista_estimados[s][i][j] = 0.01
    
# -------------- FUNCIONES IDW Y RBF ---------------#
def idw(x, y, valores, Xi, Yi):
    distancia_idw = distancia_matriz(x,y, Xi,Yi)
    # pesos = 1 / distancia
    pesos_idw = 1.0 / distancia_idw
    # suma de los pesos = 1
    pesos_idw /= pesos_idw.sum(axis=0)
    # valores observadoss multiplicados por los pesos
    z_idw = np.dot(pesos_idw.T, valores)
    return z_idw

def rbf(x, y, valores, Xi, Yi):
    distancia_rbf = distancia_matriz(x,y, Xi,Yi)
    # distancias mutuas por parejas
    distancia_int = distancia_matriz(x,y, x,y)
    # minimizar error de ajuste
    pesos_rbf = np.linalg.solve(distancia_int, valores)
    # valores observadoss multiplicados por los pesos
    z_rfb =  np.dot(distancia_rbf.T, pesos_rbf)
    return z_rfb


def scipy_idw(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='linear')
    return interpolacion(Xi, Yi)

def distancia_matriz(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interpolacion = np.vstack((x1, y1)).T
    d0 = np.subtract.outer(obs[:,0], interpolacion[:,0])
    d1 = np.subtract.outer(obs[:,1], interpolacion[:,1])
    return np.hypot(d0, d1)

def b_r_f_m(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='multiquadric')
    return interpolacion(Xi, Yi)

def b_r_f_i(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='inverse')
    return interpolacion(Xi, Yi)

def b_r_f_g(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='gaussian')
    return interpolacion(Xi, Yi)

def b_r_f_l(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='linear')
    return interpolacion(Xi, Yi)

def b_r_f_c(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='cubic')
    return interpolacion(Xi, Yi)

def b_r_f_q(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='quintic')
    return interpolacion(Xi, Yi)

def b_r_f_t(x, y, valores, Xi, Yi):
    interpolacion = Rbf(x, y, valores, function='thin_plate')
    return interpolacion(Xi, Yi)



#-EMPIEZA LO CHIDO (INTERPOLACIONES DEL DATA FRAME)-#
    # ------- CALCULO DE LOS ERRORES -------- #
estaciones = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Estaciones/Coordenadas.csv')
puntos = estaciones[['x','y']]
puntos = puntos.as_matrix()

grid_x = np.arange(-100.8,-99.9, 0.01)
grid_y = np.arange(25.3, 25.9, 0.005)
grid_y = grid_y[::-1]

Xi, Yi = np.meshgrid(grid_x, grid_y)
Xi_fun, Yi_fun = Xi.flatten(), Yi.flatten()

cc_grid_x = [55,55,46,43,34,21,46,61,70,39,81,56,49]
cc_grid_y = [45,29,45,27,43,23,19,24,48,46,107,64,34]

total_valores_interpolados = []
total_variable = []

for s in range(0,10):
    interpolaciones_vor = []
    interpolaciones_idw = []
    interpolaciones_rbf_m = []
    interpolaciones_rbf_i = []
    interpolaciones_rbf_g = []
    interpolaciones_rbf_l = []
    interpolaciones_rbf_c = []
    interpolaciones_rbf_q = []
    interpolaciones_rbf_t = []
    interpolaciones_ok = []
    interpolaciones_uk = []
    
    for i in range(len(data_seleccionados)):
        valores = total_lista_valores[s].iloc[i]
        valores = valores.as_matrix()
        
        cc_x=[]
        cc_y=[]
        for j in range(len(data_seleccionados.iloc[0])):
            if data_seleccionados[j][i]==0:
                cc_x.append(puntos[0][0])
                cc_y.append(puntos[0][1])
                #print('0')
            elif data_seleccionados[j][i]==1:
                cc_x.append(puntos[1][0])
                cc_y.append(puntos[1][1])
                #print('1')
            elif data_seleccionados[j][i]==2:
                cc_x.append(puntos[2][0])
                cc_y.append(puntos[2][1])
                #print('2')
            elif data_seleccionados[j][i]==3:
                cc_x.append(puntos[3][0])
                cc_y.append(puntos[3][1])
                #print('3')
            elif data_seleccionados[j][i]==4:
                cc_x.append(puntos[4][0])
                cc_y.append(puntos[4][1])
                #print('4')
            elif data_seleccionados[j][i]==5:
                cc_x.append(puntos[5][0])
                cc_y.append(puntos[5][1])
                #print('5')
            elif data_seleccionados[j][i]==6:
                cc_x.append(puntos[6][0])
                cc_y.append(puntos[6][1])
                #print('6')
            elif data_seleccionados[j][i]==7:
                cc_x.append(puntos[7][0])
                cc_y.append(puntos[7][1])
                #print('7')
            elif data_seleccionados[j][i]==8:
                cc_x.append(puntos[8][0])
                cc_y.append(puntos[8][1])
                #print('8')
            elif data_seleccionados[j][i]==9:
                cc_x.append(puntos[9][0])
                cc_y.append(puntos[9][1])
                #print('9')
            elif data_seleccionados[j][i]==10:
                cc_x.append(puntos[10][0])
                cc_y.append(puntos[10][1])
                #print('10')
            elif data_seleccionados[j][i]==11:
                cc_x.append(puntos[11][0])
                cc_y.append(puntos[11][1])
                #print('11')
            else:
                cc_x.append(puntos[12][0])
                cc_y.append(puntos[12][1])
                #print('12')
        
        cc_x2=[]
        cc_y2=[]
        for k in range(len(data_interpolados.iloc[0])):
            if data_interpolados[k][i]==0:
                cc_x2.append(puntos[0][0])
                cc_y2.append(puntos[0][1])
                #print('0')
            elif data_interpolados[k][i]==1:
                cc_x2.append(puntos[1][0])
                cc_y2.append(puntos[1][1])
                #print('1')
            elif data_interpolados[k][i]==2:
                cc_x2.append(puntos[2][0])
                cc_y2.append(puntos[2][1])
                #print('2')
            elif data_interpolados[k][i]==3:
                cc_x2.append(puntos[3][0])
                cc_y2.append(puntos[3][1])
                #print('3')
            elif data_interpolados[k][i]==4:
                cc_x2.append(puntos[4][0])
                cc_y2.append(puntos[4][1])
                #print('4')
            elif data_interpolados[k][i]==5:
                cc_x2.append(puntos[5][0])
                cc_y2.append(puntos[5][1])
                #print('5')
            elif data_interpolados[k][i]==6:
                cc_x2.append(puntos[6][0])
                cc_y2.append(puntos[6][1])
                #print('6')
            elif data_interpolados[k][i]==7:
                cc_x2.append(puntos[7][0])
                cc_y2.append(puntos[7][1])
                #print('7')
            elif data_interpolados[k][i]==8:
                cc_x2.append(puntos[8][0])
                cc_y2.append(puntos[8][1])
                #print('8')
            elif data_interpolados[k][i]==9:
                cc_x2.append(puntos[9][0])
                cc_y2.append(puntos[9][1])
                #print('9')
            elif data_interpolados[k][i]==10:
                cc_x2.append(puntos[10][0])
                cc_y2.append(puntos[10][1])
                #print('10')
            elif data_interpolados[k][i]==11:
                cc_x2.append(puntos[11][0])
                cc_y2.append(puntos[11][1])
                #print('11')
            else:
                cc_x2.append(puntos[12][0])
                cc_y2.append(puntos[12][1])
                #print('12')
                
        cc_nuevas=[cc_x,cc_y]
        coor_nuevas = pd.DataFrame(cc_nuevas)
        coor_nuevas=coor_nuevas.T
        coor_nuevas=coor_nuevas.as_matrix()
        
        x_selec = coor_nuevas[:, 0]
        y_selec = coor_nuevas[:, 1]
    
        cc_interp=[cc_x2,cc_y2]
        coor_interp = pd.DataFrame(cc_interp)
        coor_interp=coor_interp.T
        coor_interp=coor_interp.as_matrix() 
        
        x_interp = coor_interp[:, 0]
        y_interp = coor_interp[:, 1]
    
        #------ DIAGRAMAS DE VORONOI -----#    
        teselacion_voronoi = griddata(coor_nuevas, valores, (Xi, Yi), method='nearest')
        teselacion_voronoi = inv_boxcox(teselacion_voronoi, list_m_pearson[s]).astype('int64')
        
        valor_1 = teselacion_voronoi[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = teselacion_voronoi[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = teselacion_voronoi[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = teselacion_voronoi[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_vor = [valor_1]
        interpolaciones_vor.append(valores_vor)
        
        #-- DISTANCIA INVERSA PONDERADA --#
        dist_inv_pon = idw(x_selec, y_selec, valores,Xi_fun,Yi_fun)
        dist_inv_pon = dist_inv_pon.reshape((len(grid_y), len(grid_x)))
        dist_inv_pon = inv_boxcox(dist_inv_pon, list_m_pearson[s]).astype('int64')
        
        valor_1 = dist_inv_pon[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = dist_inv_pon[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = dist_inv_pon[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
       # valor_4 = dist_inv_pon[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_idw = [valor_1]
        interpolaciones_idw.append(valores_idw)
        
        #---- FUNCIONES DE BASE RADIAL MULTICUADRATIC---#
        fun_bas_rad_m = b_r_f_m(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_m = fun_bas_rad_m.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_m = inv_boxcox(fun_bas_rad_m, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_m[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_m[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_m[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_m[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_m = [valor_1]
        interpolaciones_rbf_m.append(valores_rbf_m)
        
        #---- FUNCIONES DE BASE RADIAL INVERSE---#
        fun_bas_rad_i = b_r_f_i(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_i = fun_bas_rad_i.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_i = inv_boxcox(fun_bas_rad_i, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_i[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_i[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_i[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_i[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_i = [valor_1]
        interpolaciones_rbf_i.append(valores_rbf_i)
        
        #---- FUNCIONES DE BASE RADIAL GAUSSIANA---#
        fun_bas_rad_g = b_r_f_g(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_g = fun_bas_rad_g.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_g = inv_boxcox(fun_bas_rad_g, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_g[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_g[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_g[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_g[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_g = [valor_1]
        interpolaciones_rbf_g.append(valores_rbf_g)
        
        #---- FUNCIONES DE BASE RADIAL LINEAL---#
        fun_bas_rad_l = b_r_f_l(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_l = fun_bas_rad_l.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_l = inv_boxcox(fun_bas_rad_l, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_l[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_l[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_l[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_l[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_l = [valor_1]
        interpolaciones_rbf_l.append(valores_rbf_l)
        
        #---- FUNCIONES DE BASE RADIAL CUBICO---#
        fun_bas_rad_c = b_r_f_c(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_c = fun_bas_rad_c.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_c = inv_boxcox(fun_bas_rad_c, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_c[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_c[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_c[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_c[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_c = [valor_1]
        interpolaciones_rbf_c.append(valores_rbf_c)
        
        #---- FUNCIONES DE BASE RADIAL QUINTIC---#
        fun_bas_rad_q = b_r_f_q(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_q = fun_bas_rad_q.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_q = inv_boxcox(fun_bas_rad_q, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_q[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_q[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_q[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_q[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_q = [valor_1]
        interpolaciones_rbf_q.append(valores_rbf_q)
        
        #---- FUNCIONES DE BASE RADIAL THIN PLATE---#
        fun_bas_rad_t = b_r_f_t(x_selec, y_selec, valores,Xi,Yi)
        fun_bas_rad_t = fun_bas_rad_t.reshape((len(grid_y), len(grid_x)))
        fun_bas_rad_t = inv_boxcox(fun_bas_rad_t, list_m_pearson[s]).astype('int64')
        
        valor_1 = fun_bas_rad_t[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = fun_bas_rad_t[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = fun_bas_rad_t[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = fun_bas_rad_t[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_rbf_t = [valor_1]
        interpolaciones_rbf_t.append(valores_rbf_t)
        
        # ----- KRIGING ORDINARIO ------ #
        OK = OrdinaryKriging(x_selec, y_selec, valores, variogram_model='linear',
                         verbose=False, enable_plotting=False)
        z_ok, ss_ok = OK.execute('grid', grid_x, grid_y)
        z_ok = inv_boxcox(z_ok, list_m_pearson[s]).astype('int64')
        
        valor_1 = z_ok[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = z_ok[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = z_ok[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = z_ok[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_ok = [valor_1]
        interpolaciones_ok.append(valores_ok)
        
        # ----- KRIGING UNIVERSAL ------ #
        UK = UniversalKriging(x_selec, y_selec, valores, variogram_model='linear',
                              drift_terms=['regional_linear'])
        z_uk, ss_uk = UK.execute('grid', grid_x, grid_y)
        z_uk = inv_boxcox(z_uk, list_m_pearson[s]).astype('int64')
        
        valor_1 = z_uk[cc_grid_y[data_interpolados[0][i]]][cc_grid_x[data_interpolados[0][i]]]
        #valor_2 = z_uk[cc_grid_y[data_interpolados[1][i]]][cc_grid_x[data_interpolados[1][i]]]
        #valor_3 = z_uk[cc_grid_y[data_interpolados[2][i]]][cc_grid_x[data_interpolados[2][i]]]
        #valor_4 = z_uk[cc_grid_y[data_interpolados[3][i]]][cc_grid_x[data_interpolados[3][i]]]
        
        valores_uk = [valor_1]
        interpolaciones_uk.append(valores_uk)
        
        print(s,i)
            
    data_int_vor = pd.DataFrame(interpolaciones_vor)
    data_int_idw = pd.DataFrame(interpolaciones_idw)
    data_int_rbf_m = pd.DataFrame(interpolaciones_rbf_m)
    data_int_rbf_i = pd.DataFrame(interpolaciones_rbf_i)
    data_int_rbf_g = pd.DataFrame(interpolaciones_rbf_g)
    data_int_rbf_l = pd.DataFrame(interpolaciones_rbf_l)
    data_int_rbf_c = pd.DataFrame(interpolaciones_rbf_c)
    data_int_rbf_q = pd.DataFrame(interpolaciones_rbf_q)
    data_int_rbf_t = pd.DataFrame(interpolaciones_rbf_t)
    data_int_ok = pd.DataFrame(interpolaciones_ok)
    data_int_uk = pd.DataFrame(interpolaciones_uk)
    
    total_variable = [data_int_vor, data_int_idw, data_int_rbf_m, data_int_rbf_i,
                      data_int_rbf_g, data_int_rbf_l, data_int_rbf_c, data_int_rbf_q,
                      data_int_rbf_t, data_int_ok, data_int_uk]
    
    total_valores_interpolados.append(total_variable)

    
#---------------------- KPI ------------------#
    # ------ CALCULO DE ERRORES ------- #
total_list_error = []
for s in range (0,10):
    list_error_variable = []
    for c in range(0,11):
        list_mae, list_mape, list_rmse, list_real = [], [], [], []
        
        for i in range(len(data_real)):
            for j in range(len(data_int_vor.loc[0])):
                real = total_lista_estimados[s][j][i]
                estimado = total_valores_interpolados[s][c][j][i]
                error = estimado-real
                
                mae = abs(error)
                rmse =  error**2
                
                list_mae.append(mae)
                list_mape.append(mae/real)
                list_rmse.append(rmse)
                list_real.append(real)
                
        print(s,c)
    
        mape = sum(list_mape)/len(list_mape)  
        mae = sum(list_mae)/len(list_mae)
        mae_p = mae/(sum(list_real)/len(list_mae))
        rmse = math.sqrt(sum(list_rmse)/len(list_rmse))
        rmse_p = rmse/(sum(list_real)/len(list_rmse))
        mse = sum(list_rmse)/len(list_rmse)
        
        list_error_variable.append([mape, mae, mae_p, rmse, rmse_p, mse])
    
    total_list_error.append(list_error_variable)
    
# -------------- IMAGENES DE LAS INTERPOLACIONES --------------- #
for s in range(0,10):
    for i in range(len(total_lista_valores[0][0])-5,len(total_lista_valores[0][0])):
        valores_figure = total_lista_valores[s].iloc[i]
        valores_figure = valores_figure.as_matrix()
        
        cc_x=[]
        cc_y=[]
        for j in range(len(data_seleccionados.iloc[0])):
            if data_seleccionados[j][i]==0:
                cc_x.append(puntos[0][0])
                cc_y.append(puntos[0][1])
            elif data_seleccionados[j][i]==1:
                cc_x.append(puntos[1][0])
                cc_y.append(puntos[1][1])
            elif data_seleccionados[j][i]==2:
                cc_x.append(puntos[2][0])
                cc_y.append(puntos[2][1])
            elif data_seleccionados[j][i]==3:
                cc_x.append(puntos[3][0])
                cc_y.append(puntos[3][1])
            elif data_seleccionados[j][i]==4:
                cc_x.append(puntos[4][0])
                cc_y.append(puntos[4][1])
            elif data_seleccionados[j][i]==5:
                cc_x.append(puntos[5][0])
                cc_y.append(puntos[5][1])
            elif data_seleccionados[j][i]==6:
                cc_x.append(puntos[6][0])
                cc_y.append(puntos[6][1])
            elif data_seleccionados[j][i]==7:
                cc_x.append(puntos[7][0])
                cc_y.append(puntos[7][1])
            elif data_seleccionados[j][i]==8:
                cc_x.append(puntos[8][0])
                cc_y.append(puntos[8][1])
            elif data_seleccionados[j][i]==9:
                cc_x.append(puntos[9][0])
                cc_y.append(puntos[9][1])
            elif data_seleccionados[j][i]==10:
                cc_x.append(puntos[10][0])
                cc_y.append(puntos[10][1])
            elif data_seleccionados[j][i]==11:
                cc_x.append(puntos[11][0])
                cc_y.append(puntos[11][1])
            else:
                cc_x.append(puntos[12][0])
                cc_y.append(puntos[12][1])
        
        cc_x2=[]
        cc_y2=[]
        for k in range(len(data_interpolados.iloc[0])):
            if data_interpolados[k][i]==0:
                cc_x2.append(puntos[0][0])
                cc_y2.append(puntos[0][1])
            elif data_interpolados[k][i]==1:
                cc_x2.append(puntos[1][0])
                cc_y2.append(puntos[1][1])
            elif data_interpolados[k][i]==2:
                cc_x2.append(puntos[2][0])
                cc_y2.append(puntos[2][1])
            elif data_interpolados[k][i]==3:
                cc_x2.append(puntos[3][0])
                cc_y2.append(puntos[3][1])
            elif data_interpolados[k][i]==4:
                cc_x2.append(puntos[4][0])
                cc_y2.append(puntos[4][1])
            elif data_interpolados[k][i]==5:
                cc_x2.append(puntos[5][0])
                cc_y2.append(puntos[5][1])
            elif data_interpolados[k][i]==6:
                cc_x2.append(puntos[6][0])
                cc_y2.append(puntos[6][1])
            elif data_interpolados[k][i]==7:
                cc_x2.append(puntos[7][0])
                cc_y2.append(puntos[7][1])
            elif data_interpolados[k][i]==8:
                cc_x2.append(puntos[8][0])
                cc_y2.append(puntos[8][1])
            elif data_interpolados[k][i]==9:
                cc_x2.append(puntos[9][0])
                cc_y2.append(puntos[9][1])
            elif data_interpolados[k][i]==10:
                cc_x2.append(puntos[10][0])
                cc_y2.append(puntos[10][1])
            elif data_interpolados[k][i]==11:
                cc_x2.append(puntos[11][0])
                cc_y2.append(puntos[11][1])
            else:
                cc_x2.append(puntos[12][0])
                cc_y2.append(puntos[12][1])
                
        cc_nuevas=[cc_x,cc_y]
        coor_nuevas = pd.DataFrame(cc_nuevas)
        coor_nuevas=coor_nuevas.T
        coor_nuevas=coor_nuevas.as_matrix()
        
        x_selec = coor_nuevas[:, 0]
        y_selec = coor_nuevas[:, 1]
    
        cc_interp=[cc_x2,cc_y2]
        coor_interp = pd.DataFrame(cc_interp)
        coor_interp=coor_interp.T
        coor_interp=coor_interp.as_matrix() 
        
        x_interp = coor_interp[:, 0]
        y_interp = coor_interp[:, 1]
    
        #------ DIAGRAMAS DE VORONOI -----#
        teselacion_voronoi = griddata(coor_nuevas, valores_figure, (Xi, Yi), method='nearest')
        
        plt.imshow(teselacion_voronoi, extent=(-100.8, -99.9,25.3, 25.9),vmin=-10, vmax=10, cmap='Greys')
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/voronoi_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
            
        #--- TRIANGULACION DE DELAUNAY ---#
        """triang_delaunay = griddata(coor_nuevas, valores_deterministico, (Xi, Yi), method='linear')
        
        plt.imshow(triang_delaunay, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/TESIS_RNA/IMAGENES/HISTORICO/TD/delaunay_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        plt.imshow(triang_delaunay, extent=(-100.8, -99.9,25.3, 25.9), cmap=mi_colormap,vmin=2,vmax=380)
        plt.plot(x_selec, y_selec,'ko', ms=3,color='black',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='black',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/TESIS_RNA/IMAGENES_SIMA/HISTORICO/TD/delaunay_sima_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()"""
        
        #-- DISTANCIA INVERSA PONDERADA --#
        dist_inv_pon = idw(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        dist_inv_pon = dist_inv_pon.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(dist_inv_pon, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/idw_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
            
        #---- FUNCIONES DE BASE RADIAL MULTICUADRATIC ---#
        fun_bas_rad_m = b_r_f_m(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_m = fun_bas_rad_m.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_m, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_m_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        #---- FUNCIONES DE BASE RADIAL INVERSE ---#
        fun_bas_rad_i = b_r_f_i(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_i = fun_bas_rad_i.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_i, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_i_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        #---- FUNCIONES DE BASE RADIAL GAUSSIANO---#
        fun_bas_rad_g = b_r_f_g(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_g = fun_bas_rad_g.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_g, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_g_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        #---- FUNCIONES DE BASE RADIAL LINEAL ---#
        fun_bas_rad_l = b_r_f_l(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_l = fun_bas_rad_l.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_l, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_l_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        #---- FUNCIONES DE BASE RADIAL CUBICO---#
        fun_bas_rad_c = b_r_f_c(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_c = fun_bas_rad_c.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_c, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_c_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        #---- FUNCIONES DE BASE RADIAL QUNTIC---#
        fun_bas_rad_q = b_r_f_q(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_q = fun_bas_rad_q.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_q, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_q_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        #---- FUNCIONES DE BASE RADIAL TPS---#
        fun_bas_rad_t = b_r_f_t(x_selec, y_selec, valores_figure,Xi_fun,Yi_fun)
        fun_bas_rad_t = fun_bas_rad_t.reshape((len(grid_y), len(grid_x)))
        
        plt.imshow(fun_bas_rad_t, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/brf_tps_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        # ----- KRIGING ORDINARIO ------ #
        OK = OrdinaryKriging(x_selec, y_selec, valores_figure, variogram_model='linear',
                         verbose=False, enable_plotting=False)
        z_ok, ss_ok = OK.execute('grid', grid_x, grid_y)
        
        plt.imshow(z_ok, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/ok_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()
        
        # ----- KRIGING UNIVERSAL ------ #
        UK = UniversalKriging(x_selec, y_selec, valores_figure, variogram_model='linear',
                              drift_terms=['regional_linear'])
        z_uk, ss_uk = UK.execute('grid', grid_x, grid_y)
        
        plt.imshow(z_uk, extent=(-100.8, -99.9,25.3, 25.9), cmap="Greys")
        plt.plot(x_selec, y_selec,'ko', ms=3,color='blue',marker="x")
        plt.plot(x_interp, y_interp, 'ko', ms=3,color='red',marker="v")
        plt.ylabel('Latitud')
        plt.xlabel('Longitud')
        plt.colorbar()
        plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/9/uk_9_'+str(s)+'_'+str(i)+'.png', format='png', dpi=1000)
        plt.show()
        plt.close()

# --- IMAGENES DE SERIES CON ATÍPICOS --- #  
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import QuantileAD

fil_var_4 = fil_var_2.copy()
for i in range(0,10):
    fil_var_4[i].index = fil_var[0].index
  
for i in range (0,10):
    serie = fil_var_4[i]
    serie = validate_series(serie)
    
    for ax in serie:
        q_a = QuantileAD(high=0.99, low=0.01)
        anomalia = q_a.fit_detect(serie)

    plot(serie, anomaly=anomalia, ts_linewidth=None, ts_markersize=0, ts_color='cyan',
         anomaly_markersize=2, anomaly_color='red', anomaly_tag="marker", legend=False)
    plt.xlabel('Tiempo')
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/serie_adtk'+str(i)+'.png', format='png', dpi=1000)
    plt.close()

# ----- BARPLOT CON PORCENTAJE DE ATIPICOS ------ #
cols_plot= ['Centro','Noreste','Noreste2','Noroeste','Noroeste2','Norte','Norte2','Sur',
            'Sureste','Sureste2', 'Sureste3', 'Suroeste','Suroeste2']

cols_var = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5",
            "SO2", "velocity", "direction"]

total_anomalias = []
for i in range(0,10):
    lista_contar_anomalias = []
    for j in range(0,13):
        serie = fil_var_4[i][j]
        serie = validate_series(serie)
        q_a = QuantileAD(high=0.99, low=0.01)
        anomalia = q_a.fit_detect(serie)
        contar_anomalia = sum(anomalia)
        
        lista_contar_anomalias.append(contar_anomalia)
    total_anomalias.append(lista_contar_anomalias)

"""for i in range(0,10):
    list_normales = list(repeat(len(list_real), 13))
    list_anomalos = total_anomalias[i]
    
    r = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    data = {'greenBars': list_normales, 'orangeBars': list_anomalos}
    df = pd.DataFrame(data)
 
    totals = [i+j for i,j in zip(df['greenBars'], df['orangeBars'])]
    greenBars = [i / j * 100 for i,j in zip(df['greenBars'], totals)]
    orangeBars = [i / j * 100 for i,j in zip(df['orangeBars'], totals)]
    barWidth = 0.85


    plt.bar(r, greenBars, color='b', edgecolor='white', width=barWidth, label=" % datos")
    plt.bar(r, orangeBars, bottom=greenBars, color='r', edgecolor='white', width=barWidth, label="% datos atípicos")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    plt.xticks(r, cols_plot, rotation=90)
    plt.xlabel(str(cols_var[i]))
    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()]) 
    plt.tight_layout()
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/BarPlot'+str(i)+'.png', format='png', dpi=1000)
    plt.show()
    plt.close()"""

# --- PIE DE ATIPICOS ---#
"""for i in range(0,10):
    fig, axs = plt.subplots(4, 4,figsize=(8,8))
    axs[0, 0].pie([list_normales[0], total_anomalias[i][0]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[0,0].set_title('Centro')
    axs[0, 1].pie([list_normales[0], total_anomalias[i][1]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[0,1].set_title('Noreste')
    axs[0, 2].pie([list_normales[0], total_anomalias[i][2]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[0,2].set_title('Noroeste 2')
    axs[0, 3].pie([list_normales[0], total_anomalias[i][3]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[0,3].set_title('Noroeste')
    axs[1, 0].pie([list_normales[0], total_anomalias[i][4]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[1,0].set_title('Noroeste 2')
    axs[1, 1].pie([list_normales[0], total_anomalias[i][5]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[1,1].set_title('Norte')
    axs[1, 2].pie([list_normales[0], total_anomalias[i][6]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[1,2].set_title('Norte 2')
    axs[1, 3].pie([list_normales[0], total_anomalias[i][7]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[1,3].set_title('Sur')
    axs[2, 0].pie([list_normales[0], total_anomalias[i][8]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[2,0].set_title('Sureste')
    axs[2, 1].pie([list_normales[0], total_anomalias[i][9]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[2,1].set_title('Sureste 2')
    axs[2, 2].pie([list_normales[0], total_anomalias[i][10]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[2,2].set_title('Sureste 3')
    axs[2, 3].pie([list_normales[0], total_anomalias[i][11]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[2,3].set_title('Suroeste')
    axs[3, 0].pie([list_normales[0], total_anomalias[i][12]], autopct='%1.1f%%', shadow=False, colors=("b","r"), radius=1.4)
    axs[3,0].set_title('Suroeste 2')
    fig.delaxes(axs[3][1])
    fig.delaxes(axs[3][2])
    fig.delaxes(axs[3][3])
    plt.tight_layout()
    plt.suptitle(str(cols_var[i]))
    plt.savefig('C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/Pie_'+str(i)+'.png', format='png', dpi=1000)
    plt.show()
    plt.close()"""




