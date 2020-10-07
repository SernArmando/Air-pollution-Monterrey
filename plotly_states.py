# -*- coding: utf-8 -*-
"""
Created 2020

@author: Serna
"""
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.offline import plot

mi_colormap = [0,"rgb(76,255,44)"],[61/380,"rgb(255,234,44)"],\
[121/380,"rgb(255,132,44)"], [221/380,"rgb(255,44,44)"], [1,"rgb(100,44,255)"]

dstates = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_estados.csv').query("id == 19")
dm2 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19004")
dm3 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19006")
dm4 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19009")
dm5 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19018")
dm6 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19019")
dm7 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19021")
dm8 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19026")
dm9 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19031")
dm10 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19034")
dm11 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19039")
dm12 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19041")
dm13 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19048")
dm14 = pd.read_csv('C:/Users/arman/Documents/Tesis_Mae/Datos/Mapas_Mexico/mapa_municipios.csv').query("id == 19049")

dstates=dstates.reset_index()
dm2=dm2.reset_index()
dm3=dm3.reset_index()
dm4=dm4.reset_index()
dm5=dm5.reset_index()
dm6=dm6.reset_index()
dm7=dm7.reset_index()
dm8=dm8.reset_index()
dm9=dm9.reset_index()
dm10=dm10.reset_index()
dm11=dm11.reset_index()
dm12=dm12.reset_index()
dm13=dm13.reset_index()
dm14=dm14.reset_index()


Xi_map = pd.Series(Xi_fun)
Yi_map = pd.Series(Yi_fun)
tes_vor_map = z_ok.flatten()
tes_vor_map = pd.Series(tes_vor_map)

fig = go.Figure(data=go.Scattergeo(lat = Yi_map, lon = Xi_map,text = tes_vor_map.astype(str) + ' u/gr de contaminante PM10',
    marker = dict(color = tes_vor_map, colorscale = mi_colormap, symbol=1,reversescale = False, opacity = 0.7, size = 5,
    cmax = 380, cmin = 2,colorbar = dict(titleside = "right", outlinecolor = "rgba(68, 68, 68, 0)",ticks = "", showticksuffix = "none",
    dtick = 10))))

for i in range(len(dstates)-1):
    fig.add_trace(go.Scattergeo(lon = [dstates['long'][i], dstates['long'][i+1]],lat = [dstates['lat'][i], dstates['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm2)-1):
    fig.add_trace(go.Scattergeo(lon = [dm2['long'][i], dm2['long'][i+1]],lat = [dm2['lat'][i], dm2['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm3)-1):
    fig.add_trace(go.Scattergeo(lon = [dm3['long'][i], dm3['long'][i+1]],lat = [dm3['lat'][i], dm3['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm4)-1):
    fig.add_trace(go.Scattergeo(lon = [dm4['long'][i], dm4['long'][i+1]],lat = [dm4['lat'][i], dm4['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))
    
for i in range(len(dm5)-1):
    fig.add_trace(go.Scattergeo(lon = [dm5['long'][i], dm5['long'][i+1]],lat = [dm5['lat'][i], dm5['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm6)-1):
    fig.add_trace(go.Scattergeo(lon = [dm6['long'][i], dm6['long'][i+1]],lat = [dm6['lat'][i], dm6['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm7)-1):
    fig.add_trace(go.Scattergeo(lon = [dm7['long'][i], dm7['long'][i+1]],lat = [dm7['lat'][i], dm7['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm8)-1):
    fig.add_trace(go.Scattergeo(lon = [dm8['long'][i], dm8['long'][i+1]],lat = [dm8['lat'][i], dm8['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm9)-1):
    fig.add_trace(go.Scattergeo(lon = [dm9['long'][i], dm9['long'][i+1]],lat = [dm9['lat'][i], dm9['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm10)-1):
    fig.add_trace(go.Scattergeo(lon = [dm10['long'][i], dm10['long'][i+1]],lat = [dm10['lat'][i], dm10['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm11)-1):
    fig.add_trace(go.Scattergeo(lon = [dm11['long'][i], dm11['long'][i+1]],lat = [dm11['lat'][i], dm11['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm12)-1):
    fig.add_trace(go.Scattergeo(lon = [dm12['long'][i], dm12['long'][i+1]],lat = [dm12['lat'][i], dm12['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm13)-1):
    fig.add_trace(go.Scattergeo(lon = [dm13['long'][i], dm13['long'][i+1]],lat = [dm13['lat'][i], dm13['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))

for i in range(len(dm14)-1):
    fig.add_trace(go.Scattergeo(lon = [dm14['long'][i], dm14['long'][i+1]],lat = [dm14['lat'][i], dm14['lat'][i+1]],mode = 'lines', line = dict(width = 1,color = 'black')))
     
fig.update_layout(geo = dict(scope = 'world', showland = True, landcolor = "rgb(212, 212, 212)",subunitcolor = "rgb(255, 255, 255)", countrycolor = "rgb(255, 255, 255)",showlakes = True, lakecolor = "rgb(141,213,227)",
    showocean=True, oceancolor="rgb(141,213,227)",showrivers=True, rivercolor="rgb(141,213,227)",showsubunits = True, showcountries = True, resolution = 50,projection = dict(type = "orthographic", rotation_lon = -100),
    lonaxis = dict(showgrid = True, gridwidth = 0.5,range= [ -108.0,-95.0 ], dtick = 5),lataxis = dict (showgrid = True, gridwidth = 0.5,range= [ 22.0, 28.0 ], dtick = 5)),
    title='Nivel de contaminación PM10 (Nuelo León) '+str(i)+'<br>Fuente: <a href="http://aire.nl.gob.mx/map_calidad.html">SIMA</a>', showlegend = False)

fig.show()
plot(fig, filename = 'C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/KO.html', auto_open=False)


# --------------- FIGURAS TESIS -------------- #
figure = px.scatter_mapbox(z_ok, lat=Yi_map, lon=Xi_map,color=tes_vor_map,color_continuous_scale=mi_colormap,
                        size_max=15, zoom=9, text = tes_vor_map.astype(str) + ' u/gr de contaminante PM10',
                        opacity = 0.35, mapbox_style="stamen-terrain")
figure.update_layout(title='Nivel de contaminación PM10 (Nuelo León) '+str(i)+'<br>Fuente: <a href="http://aire.nl.gob.mx/map_calidad.html">SIMA</a>', showlegend = False)
figure.update_layout(coloraxis_colorbar=dict(title="Índice Aire y Salud",
    tickvals=[13,51,76,116,170],
    ticktext=["Buena", "Aceptable", "Mala", "Muy Mala","Extremadamente Mala"],))
figure.show()
plotly.offline.plot(figure, filename=f'C:/Users/arman/Documents/Tesis_Mae/Imagenes/Imagenes_Sima/ejemplo2.html')


"open-street-map", "stamen-terrain",


