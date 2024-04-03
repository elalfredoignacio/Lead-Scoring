#!/usr/bin/env python
# coding: utf-8

# # CODIGO DE EJECUCION

# In[1]:


import cloudpickle
import pandas as pd

ruta_proyecto = 'C:/Usuarios/Alfredo/DS4B/Python DS Mastery/EstructuraDirectorio/03_MACHINE_LEARNING/07_CASOS/01_LEADSCORING'

nombre_fichero_datos = 'validacion.csv' #es un dataset con nuevos datos (no confundir con validation)

ruta_completa = ruta_proyecto + '/02_Datos/02_Validacion/' + nombre_fichero_datos

df = pd.read_csv(ruta_completa,index_col='id')

df.drop_duplicates(inplace = True)
df = df.loc[(df.no_llamar != 'OTROS') & (df.no_enviar_email != 'Yes') & (df.ult_actividad != 'Email Bounced')]
                     
variables_finales = ['ambito',
                   'descarga_lm',
                   'ocupacion',
                   'paginas_vistas_visita',
                   'score_actividad',
                   'score_perfil',
                   'tiempo_en_site_total',
                   'ult_actividad',
                   'visitas_total'
                  ]

df = df[variables_finales]

nombre_pipe_ejecucion = 'pipe_ejecucion.pickle'

ruta_pipe_ejecucion = ruta_proyecto + '/04_Modelos/' + nombre_pipe_ejecucion

with open(ruta_pipe_ejecucion, mode='rb') as file:
   pipe_ejecucion = cloudpickle.load(file)

scoring = pipe_ejecucion.predict_proba(df)[:, 1]

