#!/usr/bin/env python
# coding: utf-8

# In[1]:


#la referencia es la profundiad del techo de la cámara con respecto a la superficie (4.3 km)


# In[2]:


radio = 6
y_surf = 4.3
TOP_DEPTH_LIST_KM = [2.0, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

tope_camara = TOP_DEPTH_LIST_KM[3]

disk1_prof = round(y_surf - tope_camara,2)
disk2_prof = round(disk1_prof - 4.0,2)

print("techo de camara =", tope_camara, "-> disk1:", disk1_prof, " disk2:", disk2_prof)

cham_temp = 850

figs = False


# In[3]:


import os
os.environ["OMPI_MCA_btl"] = "self,tcp"
os.environ["OMPI_MCA_pml"] = "ob1"
os.environ["OMPI_MCA_mtl"] = "^ofi"     # desactiva el MTL ofi
os.environ["FI_PROVIDER"]   = "tcp"     # libfabric via tcp
os.environ["UCX_TLS"]       = "tcp"     # por si está UCX
# opcional: evita que PSM3 intente pinnear CPUs/NICs
os.environ["PSM3_ALLOW_ROUND_ROBIN_CPUS"] = "1"


# In[4]:


import underworld as uw
import underworld.function as fn
from underworld import UWGeodynamics as GEO
import numpy as np
import math
import random
import time
import csv
import shutil
from numpy import loadtxt
import matplotlib.pyplot as plt
from underworld import visualisation as vis
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm


import warnings
import matplotlib.cbook
#warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# In[5]:


u = GEO.UnitRegistry


# ## Scaling

# In[6]:


half_rate     = 0.25 * u.centimeter / u.year
model_height  = 29.0 * u.km   # (2 air + 2 stickyAir)
surfaceTemp   = 293.15 * u.degK
baseModelTemp = 813.15 * u.degK
bodyforce     = 2750 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_height
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2
KT = (baseModelTemp - surfaceTemp)

GEO.scaling_coefficients["[length]"]      = KL
GEO.scaling_coefficients["[time]"]        = Kt
GEO.scaling_coefficients["[mass]"]        = KM
GEO.scaling_coefficients["[temperature]"] = KT


# ## Resolución

# In[7]:


dx = 0.200 * u.km
dy = 0.200 * u.km

maxx = 30 * u.km
maxy = 8.3 * u.km

minx = 0.0   * u.km
miny = maxy - 34.0 * u.km


# In[8]:


Lx = (maxx - minx)
Lz = (maxy - miny)

# Si querés potencias de 2, activá el helper de abajo:
def next_pow2(n):
    import math
    return 1 << (int(math.ceil(math.log(max(n,1), 2))))

# Opción A: resolución exacta por tamaño de celda
resX_exact = int(np.ceil(GEO.non_dimensionalise(Lx/dx)))
resY_exact = int(np.ceil(GEO.non_dimensionalise(Lz/dy)))

# Opción B (recomendada para estabilidad y rendimiento): redondear a potencia de 2
resX = next_pow2(resX_exact)   # p.ej., 32 o 64
resY = next_pow2(resY_exact)   # p.ej., 128 o 256

print("resX_exact, resY_exact =", resX_exact, resY_exact)
print("resX(pow2), resY(pow2) =", resX, resY)
print("Δx ~", GEO.non_dimensionalise(Lx/resX), "km")
print("Δz ~", GEO.non_dimensionalise(Lz/resY), "km")


# In[9]:


# --- Crear carpeta de salida ---
output_dir = "R_" + str(radio) + "_steady-state_" + str(resX) + 'x' +str(resY)
os.makedirs(output_dir, exist_ok=True)


# ## Modelo

# In[10]:


Model = GEO.Model(elementRes=(resX,resY), 
                  minCoord=(minx, miny), 
                  maxCoord=(maxx, maxy), 
                  gravity=(0.0, -9.81 * u.meter / u.second**2))


# In[11]:


reGrid = False


# In[12]:


if reGrid == True:
    beta = 0.5               # <1 concentra arriba; >1 concentra abajo
    am, bm = Model.mesh.minCoord[1], Model.mesh.maxCoord[1]

    y0 = (Model.mesh.data[:,1] - am) / (bm - am)    # [0,1] bottom→top
    y1 = y0**beta                                   # power-law
    y_new = am + y1 * (bm - am)

    with Model.mesh.deform_mesh():
        Model.mesh.data[:,1] = y_new


# In[13]:


if reGrid == True:
    figMesh = vis.Figure(figsize=(1200,800))
    figMesh.append(vis.objects.Mesh(Model.mesh))
    figMesh.show()


# ## Diffusivity

# In[14]:


Model.diffusivity = 1.4e-6 * u.metre**2 / u.second  #1e-7 * u.metre**2 / u.second 
Model.capacity    = 1000. * u.joule / (u.kelvin * u.kilogram) # volumetric specific heat = 2.65 x106 J K-1m-3

#conductivity = Model.diffusivity * Model.capacity * 2750 * u.kilogram / u.metre**3


# In[15]:


# Top y bottom del dominio
surf       = Model.top

# Interfaces estratigráficas definidas desde la superficie (maxy)
z_dep_superf = maxy - (0.2  + 4.0) * u.km   # Depósitos superficiales (0–0.2 km)
z_cap_sup    = maxy - (0.8  + 4.0) * u.km   # Ignimbrita porosa (0.2–0.8 km)
z_cap_inf    = maxy - (1.5  + 4.0) * u.km   # Ignimbrita densa (0.8–1.5 km)
z_reservoir  = maxy - (3.5  + 4.0) * u.km   # Reservorio geotermal (1.5–3.5 km)
z_GrGran     = maxy - (6.5  + 4.0) * u.km   # Granito–granodiorita (3.5–6.5 km)
z_trans      = maxy - (10.0 + 4.0) * u.km   # Zona transicional/máfica (6.5–10 km)
z_andesite   = maxy - (14.0 + 4.0) * u.km   # Zona de recarga andesítica (10–14 km)
z_basement   = maxy - (30.0 + 4.0) * u.km   # Basamento metamórfico (14–30 km)


# In[16]:


# =========================
# Capas artificiales
# =========================
air = Model.add_material(
    name  = "Air",
    shape = GEO.shapes.Layer(top=surf, bottom=surf - 2.0 * u.km)
)

stickyAir = Model.add_material(
    name  = "StickyAir",
    shape = GEO.shapes.Layer(top=air.bottom, bottom=air.bottom - 2.0 * u.km)
)


# In[17]:


# =========================
# Capas geológicas
# =========================
dep_superf = Model.add_material(
    name  = "Dep_Superficiales",
    shape = GEO.shapes.Layer(top=stickyAir.bottom, bottom=z_dep_superf)
)

cap_porosa = Model.add_material(
    name  = "Cap_Rock_Porosa",
    shape = GEO.shapes.Layer(top=dep_superf.bottom, bottom=z_cap_sup)
)

cap_densa = Model.add_material(
    name  = "Cap_Rock_Densa",
    shape = GEO.shapes.Layer(top=cap_porosa.bottom, bottom=z_cap_inf)
)

reservorio = Model.add_material(
    name  = "Reservorio_Geotermal",
    shape = GEO.shapes.Layer(top=cap_densa.bottom, bottom=z_reservoir)
)

granito_granodiorita = Model.add_material(
    name  = "Granito_Granodiorita",
    shape = GEO.shapes.Layer(top=reservorio.bottom, bottom=z_GrGran)
)

zona_transicional = Model.add_material(
    name  = "Zona_Transicional_Mafica",
    shape = GEO.shapes.Layer(top=granito_granodiorita.bottom, bottom=z_trans)
)

recarga_andesitica = Model.add_material(
    name  = "Recarga_Andesitica",
    shape = GEO.shapes.Layer(top=zona_transicional.bottom, bottom=z_andesite)
)

basamento = Model.add_material(
    name  = "Basamento_Metamorfico",
    shape = GEO.shapes.Layer(top=recarga_andesitica.bottom, bottom=z_basement)
)
#Con esto para generar la geometría

disk1 = GEO.shapes.Disk(center=(maxx/2, disk1_prof * u.km),
                        radius = radio * u.km)

disk2 = GEO.shapes.Disk(center=(maxx/2, disk2_prof * u.km),
                        radius = radio * u.km)

shape = disk1 & disk2

magma_chamber = Model.add_material(name="Material", shape=shape)


# In[18]:


if figs == True:

    colormaps = ['diverge', 'isolum', 'isorainbow', 'cubelaw', 'cubelaw2', 'smoothheat', 'coolwarm', 'spectral', 'drywet', 'elevation', 'dem1', 'dem2', 'dem3', 'dem4', 'ocean', 'bathy', 'seafloor', 'abyss', 'ibcso', 'gebco', 'topo', 'sealand', 'nighttime', 'world', 'geo', 'terra', 'relief', 'globe', 'earth', 'etopo1', 'cubhelix', 'hot', 'cool', 'copper', 'gray', 'split', 'polar', 'red2green', 'paired', 'categorical', 'haxby', 'jet', 'panoply', 'no_green', 'wysiwyg', 'seis', 'rainbow', 'nih']

    mat_labels = [
        "Air",
        "StickyAir",
        "Depósitos superficiales",
        "Cap porosa",
        "Cap densa",
        "Reservorio",
        "Granito–granodiorita",
        "Transicional/máfica",
        "Recarga andesítica",
        "Basamento metamórfico", 
        ]

    Fig = vis.Figure(figsize=(1200,800),
                     title="Material Field at " + str(Model.time),
                     quality=4)

    Fig.Points(Model.swarm,
               Model.materialField,
               fn_size=4.0,
               colours = colormaps[40],
               discrete = True)

    cb = Fig.objects[0].colourBar
    cb["ticks"]       = len(mat_labels)
    cb["binlabels"]   = mat_labels  
    cb["align"]       = "right"

    Fig.axisLabels = ["X (km)", "Z (km)"]

    Fig.show()


# ## Densities

# In[19]:


alpha = 0 / u.kelvin
betas = 0 /u.pascal


# In[20]:


air.density                  = 1. * u.kilogram / u.metre**3
stickyAir.density            = 1. * u.kilogram / u.metre**3
dep_superf.density           = GEO.LinearDensity(1200. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # depósitos sueltos / ceniza
cap_porosa.density           = GEO.LinearDensity(2000. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # ignimbrita porosa/alterada (cap sup)
cap_densa.density            = GEO.LinearDensity(2300. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # ignimbrita densamente soldada (cap inf)
reservorio.density           = GEO.LinearDensity(2690. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # granito/metased. fracturado (reservorio)
granito_granodiorita.density = GEO.LinearDensity(2690. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # plutón superior
zona_transicional.density    = GEO.LinearDensity(2900. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # transición/máfica (plutón inferior)
recarga_andesitica.density   = GEO.LinearDensity(2750. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # zona de recarga andesítica
basamento.density            = GEO.LinearDensity(2920. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)  # basamento metamórfico profundo

magma_chamber.density        = GEO.LinearDensity(2450. * u.kilogram / u.metre**3, thermalExpansivity = alpha, beta = betas)
#Acá agregar la densidad como material.density, más siliceo para tener composición riolítica (asumo)


# ## Viscosities
# 

# In[21]:


rh = GEO.ViscousCreepRegistry()


# In[22]:


air.viscosity                = 1e19 * u.pascal * u.second
stickyAir.viscosity          = 1e20 * u.pascal * u.second


# In[23]:


Model.minViscosity = 1e18 * u.pascal * u.second
Model.maxViscosity = 1e25 * u.pascal * u.second


# In[24]:


dep_superf.viscosity           = rh.Wet_Quartz_Dislocation_Gleason_and_Tullis_1995           # depósitos volcánicos poco consolidados
cap_porosa.viscosity           = rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990            # ignimbrita porosa (alterada, con fluidos)
cap_densa.viscosity            = rh.Dry_Quartz_Dislocation_Brace_and_Kohlstedt_1980          # ignimbrita densamente soldada
reservorio.viscosity           = rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990            # reservorio granítico fracturado con fluidos
granito_granodiorita.viscosity = rh.Dry_Quartz_Dislocation_Koch_et_al_1983                   # plutón granítico-granodiorítico
zona_transicional.viscosity    = rh.Wet_Anorthite_Dislocation_Ribacki_et_al_2000 #rh.Dry_Maryland_Diabase_Dislocation_Mackwell_et_al_1998     # transición máfica (gabbro/diabase)
recarga_andesitica.viscosity   = rh.Wet_Anorthite_Dislocation_Ribacki_et_al_2000             # zona andesítica (feldespato cálcico)
basamento.viscosity            = rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012          # basamento metamórfico máfico-félsico
magma_chamber.viscosity        = 1e19 * u.pascal * u.second #0.1 * rh.Dry_Olivine_Dislocation_Hirth_and_Kohlstedt_2003

#material.viscosity, acá sería según las inclusiones fluidas y la temperatura? si los productos son siliceos debería tener viscosidad alta?


# In[25]:


dep_superf.viscosity.waterFugacity = 1.0e9 * u.pascal
reservorio.viscosity.waterFugacity = 1.0e9 * u.pascal
recarga_andesitica.viscosity.waterFugacity = 1.0e9 * u.pascal

dep_superf.viscosity.waterFugacityExponent = 1
reservorio.viscosity.waterFugacityExponent = 1
recarga_andesitica.viscosity.waterFugacityExponent = 1

#magma_chamber.viscosity.waterFugacity = 1.5e8 * u.pascal
#magma_chamber.viscosity.waterFugacityExponent = 1.2


# In[26]:


##para probar:

#magma_chamber.viscosity = rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
#magma_chamber.viscosity.grainSize = 2e-5 * u.metre   # si querés dependencia
#magma_chamber.viscosity.grainSizeExponent = 0        # o 1–2 si modelás pinning
#magma_chamber.viscosity.waterFugacity = 1e8 * u.pascal
#magma_chamber.viscosity.waterFugacityExponent = 1


# In[27]:


dep_superf.viscosity.grainSize            = 1.0e-5  * u.metre    # heredado de dep_intra
cap_porosa.viscosity.grainSize            = 8.0e-5  * u.metre    # cap porosa
cap_densa.viscosity.grainSize             = 8.0e-5  * u.metre    # cap densa (podés bajar a ~6e-5 si querés más “rígida”)
reservorio.viscosity.grainSize            = 2.0e-4  * u.metre    # heredado de hydro_reservoir
granito_granodiorita.viscosity.grainSize  = 5.0e-4  * u.metre    # heredado de granite_basement
zona_transicional.viscosity.grainSize     = 1.0e-3  * u.metre    # heredado de diabase/sill
recarga_andesitica.viscosity.grainSize    = 2.0e-5  * u.metre    # mapeado desde “mush” (ajustable)
basamento.viscosity.grainSize             = 1.0e-3  * u.metre    # heredado de zona MASH

dep_superf.viscosity.grainSizeExponent = 0
reservorio.viscosity.grainSizeExponent = 0
recarga_andesitica.viscosity.grainSizeExponent = 0


# ## Propiedades térmicas

# In[28]:


dep_superf.radiogenicHeatProd           = 1.1e-6 * u.watt / u.metre**3   # depósitos sueltos / ceniza
cap_porosa.radiogenicHeatProd           = 2.0e-6 * u.watt / u.metre**3   # ignimbrita porosa (límite bajo)
cap_densa.radiogenicHeatProd            = 2.9e-6 * u.watt / u.metre**3   # ignimbrita densa (valor medio)
reservorio.radiogenicHeatProd           = 6.2e-6 * u.watt / u.metre**3   # granito / metased. fracturado
granito_granodiorita.radiogenicHeatProd = 6.2e-6 * u.watt / u.metre**3   # plutón superior (igual al reservorio)
zona_transicional.radiogenicHeatProd    = 1.0e-7 * u.watt / u.metre**3   # transición máfica (muy bajo, tipo diabase/basalto)
recarga_andesitica.radiogenicHeatProd   = 4.0e-6 * u.watt / u.metre**3   # zona de recarga andesítica (valor intermedio tipo “mush”)
basamento.radiogenicHeatProd            = 1.4e-6 * u.watt / u.metre**3   # basamento metamórfico profundo

magma_chamber.radiogenicHeatProd        = 1.0e-6 * u.watt / u.metre**3   


# In[29]:


#https://pubs.usgs.gov/of/1988/0441/report.pdf
#https://link.springer.com/article/10.1023/B:NARR.0000032647.41046.e7


# In[30]:


air.diffusivity = 1.0e-6 * u.metre**2 / u.second
stickyAir.diffusivity = 1.0e-6 * u.metre**2 / u.second


# In[31]:


dep_superf.diffusivity           = 5.0e-7  * u.metre**2 / u.second   # depósitos sueltos / ceniza (clay soil)
cap_porosa.diffusivity           = 7.0e-7  * u.metre**2 / u.second   # ignimbrita porosa (ligeramente menor que la densa)
cap_densa.diffusivity            = 8.0e-7  * u.metre**2 / u.second   # ignimbrita densamente soldada (welded tuff)
reservorio.diffusivity           = 1.6e-6  * u.metre**2 / u.second   # granito/metased. fracturado (granite)
granito_granodiorita.diffusivity = 1.6e-6  * u.metre**2 / u.second   # plutón granítico-granodiorítico (granite)
zona_transicional.diffusivity    = 9.0e-7  * u.metre**2 / u.second   # transición máfica (basalt/diabase)
recarga_andesitica.diffusivity   = 1.4e-6  * u.metre**2 / u.second   # recarga andesítica (riolita/mush)
basamento.diffusivity            = 1.2e-6  * u.metre**2 / u.second   # basamento metamórfico profundo (gabbro/granulita)

magma_chamber.diffusivity        = 5.0e-7  * u.metre**2 / u.second   


# In[32]:


# Depósitos superficiales
dep_superf.capacity = 900. * u.joule / (u.kilogram * u.kelvin)    
# Ceniza / suelo volcánico fino: 820–1000 (Waples & Waples 2004)

# Cap porosa (ignimbrita poco consolidada)
cap_porosa.capacity = 880. * u.joule / (u.kilogram * u.kelvin)   
# intermedio entre porosa y densa

# Cap densa (ignimbrita densamente soldada — riolita)
cap_densa.capacity = 860. * u.joule / (u.kilogram * u.kelvin)    
# Robertson 1988, Table 12 (820–900)

# Reservorio (granito / metasedimentos)
reservorio.capacity = 790. * u.joule / (u.kilogram * u.kelvin)   
# granito fresco 770–800 (Waples & Waples 2004)

# Granito–granodiorita
granito_granodiorita.capacity = 790. * u.joule / (u.kilogram * u.kelvin) 
# mismo valor que granito fresco

# Zona transicional/máfica (basalto/diabase)
zona_transicional.capacity = 850. * u.joule / (u.kilogram * u.kelvin)  
# Robertson 1988, Table 12 (840–900)

# Recarga andesítica (mush intermedio riolita–andesita)
recarga_andesitica.capacity = 1000. * u.joule / (u.kilogram * u.kelvin) 
# mezcla parcial: cristales 800 + fundido silícico ~1500

# Basamento metamórfico profundo (gabbro/granulita)
basamento.capacity = 830. * u.joule / (u.kilogram * u.kelvin)    
# Waples & Waples 2004 (810–850)

magma_chamber.capacity = 1200. * u.joule / (u.kilogram * u.kelvin) 


# In[33]:


conductivity_dep_superf = dep_superf.diffusivity * dep_superf.capacity * 1200. * u.kilogram / u.metre**3
print("k_dep_superf = ", round(conductivity_dep_superf.to_base_units().magnitude, 2))

conductivity_cap_porosa = cap_porosa.diffusivity * cap_porosa.capacity * 2000. * u.kilogram / u.metre**3
print("k_cap_porosa = ", round(conductivity_cap_porosa.to_base_units().magnitude, 2))

conductivity_cap_densa = cap_densa.diffusivity * cap_densa.capacity * 2300. * u.kilogram / u.metre**3
print("k_cap_densa = ", round(conductivity_cap_densa.to_base_units().magnitude, 2))

conductivity_reservorio = reservorio.diffusivity * reservorio.capacity * 2690. * u.kilogram / u.metre**3
print("k_reservorio = ", round(conductivity_reservorio.to_base_units().magnitude, 2))

conductivity_granito = granito_granodiorita.diffusivity * granito_granodiorita.capacity * 2690. * u.kilogram / u.metre**3
print("k_granito_granodiorita = ", round(conductivity_granito.to_base_units().magnitude, 2))

conductivity_transicional = zona_transicional.diffusivity * zona_transicional.capacity * 2900. * u.kilogram / u.metre**3
print("k_zona_transicional = ", round(conductivity_transicional.to_base_units().magnitude, 2))

conductivity_andesita = recarga_andesitica.diffusivity * recarga_andesitica.capacity * 2750. * u.kilogram / u.metre**3
print("k_recarga_andesitica = ", round(conductivity_andesita.to_base_units().magnitude, 2))

conductivity_basamento = basamento.diffusivity * basamento.capacity * 2920. * u.kilogram / u.metre**3
print("k_basamento = ", round(conductivity_basamento.to_base_units().magnitude, 2))


# ### Plasticity

# In[34]:


pl = GEO.PlasticityRegistry()


# In[35]:


dep_superf.plasticity           = pl.Rey_et_al_2014_UpperCrust
cap_porosa.plasticity           = pl.Rey_et_al_2014_UpperCrust
cap_densa.plasticity            = pl.Rey_et_al_2014_UpperCrust
reservorio.plasticity           = pl.Rey_and_Muller_2010_UpperCrust
granito_granodiorita.plasticity = pl.Rey_and_Muller_2010_UpperCrust
zona_transicional.plasticity    = pl.Rey_and_Muller_2010_UpperCrust
recarga_andesitica.plasticity   = pl.Rey_and_Muller_2010_UpperCrust
basamento.plasticity            = pl.Rey_et_al_2014_LowerCrust


# ### HeatFlow

# In[36]:


Model.set_heatFlowBCs(bottom = (80. * u.milliwatt / u.metre**2, basamento))


# ## Temperature Boundary Conditions

# In[37]:


Tsup = 293 * u.degK


# In[38]:


Model.set_temperatureBCs(top = Tsup, 
                         bottom = None, 
                        materials=[(air, Tsup), (stickyAir, Tsup)])


# ## Velocity Boundary Conditions

# In[39]:


Model.set_velocityBCs(left = [0, None],
                      right = [0, None],
                      bottom = [None, 0.], #GEO.LecodeIsostasy(reference_mat=uppercrust,average=True),
                      top = [None, 0.])


# In[40]:


#Model.freeSurface = True


# In[41]:


u = GEO.u

# --- 0) Parámetros base ---
# Temperatura superficial (convierte °C -> K si llega sin unidades)
T0 = (Tsup.to(u.degK) if hasattr(Tsup, "to") else (Tsup + 273.15) * u.degK)

# Flujo basal de referencia (solo informativo aquí; no se usa directo en el ensamblado)
q0_mWm2 = 80.0 * u.milliwatt / u.metre**2
q0      = q0_mWm2.to(u.watt / u.kilometer**2)

# Conductividad y radiogénesis típicas de corteza (por si las querés usar luego)
k_SI = 2.5 * u.watt / (u.meter * u.degK)
k    = k_SI.to(u.watt / (u.kilometer * u.degK))
A_SI = 1.5e-6 * u.watt / (u.meter**3)
A    = A_SI.to(u.watt / u.kilometer**3)

# Geometría térmica mayor
Hc    = 18.0 * u.kilometer           # límite corteza superior
H_LAB = 40.0 * u.kilometer           # LAB efectivo
T_LAB = (900.0 + 273.15) * u.degK    # temperatura al LAB
m2    = 0.30 * u.degK / u.kilometer  # gradiente mantélico (si extendés > LAB)

# (Cálculo “teórico” de Tc y m1 por continuidad de flujo y radiogénesis;
#  queda de referencia, NO lo usamos para ensamblar el geotermo)
Tc_teor = T0 + (q0 * Hc) / k - (A * Hc**2) / (2.0 * k)
m1_teor = (T_LAB - Tc_teor) / (H_LAB - Hc)

# --- 1) Profundidad positiva hacia abajo (km) ---
# z = 0 en superficie; z > 0 hacia abajo (en km, no dimensional)
z = GEO.nd(maxy) - Model.y

# --- 2) Quiebres en la corteza (0–Hc): 5 y 30 km ---
z1 = 5.0  * u.kilometer
z2 = 30.0 * u.kilometer

# Gradientes por tramo (ajustables)
#   0–5  km : G0  (ej. 25–30 °C/km)
#   5–30 km : G1  (ej. 35–45 °C/km)
#   30–Hc   : G2  (ej. 15–25 °C/km)
G0 = 25.0 * u.degK / u.kilometer
G1 = 30.0 * u.degK / u.kilometer
G2 = 25.0 * u.degK / u.kilometer

# Temperaturas de anclaje internas (garantizan continuidad en 5 y 30 km)
Tz1 = GEO.nd(T0) + GEO.nd(G0) * GEO.nd(z1)                         # T(5 km)
Tz2 = Tz1 + GEO.nd(G1) * (GEO.nd(z2) - GEO.nd(z1))                 # T(30 km)

# Tramos corteza
T_0_z1  = GEO.nd(T0) + GEO.nd(G0) * z                              # 0–5
T_z1_z2 = Tz1 + GEO.nd(G1) * (z - GEO.nd(z1))                      # 5–30
T_z2_Hc = Tz2 + GEO.nd(G2) * (z - GEO.nd(z2))                      # 30–Hc

# --- 3) Litosfera (Hc–LAB) con CONTINUIDAD exacta en Hc ---
# Temperatura en Hc que trae la corteza:
Tc_cont = Tz2 + GEO.nd(G2) * (GEO.nd(Hc) - GEO.nd(z2))
# Pendiente litosférica que conecta Tc_cont con T_LAB:
m1 = (GEO.nd(T_LAB) - Tc_cont) / (GEO.nd(H_LAB) - GEO.nd(Hc))
# Geotermo litosférico:
lithosphere_geotherm = GEO.nd(m1) * (z - GEO.nd(Hc)) + Tc_cont

# --- 4) Manto (>LAB), si tu dominio llega más profundo que el LAB ---
mantle_geotherm = GEO.nd(m2) * (z - GEO.nd(H_LAB)) + GEO.nd(T_LAB)

# --- 5) Ensamble piecewise en una sola función: geotherm_fn ---
# Corteza 0–Hc con quiebres
crust_piecewise = fn.branching.conditional([
    (z >= GEO.nd(z2), T_z2_Hc),   # 30–Hc
    (z >= GEO.nd(z1), T_z1_z2),   # 5–30
    (True,            T_0_z1),    # 0–5
])

# Geotermo completo
T_air = Tsup.to(u.degK).magnitude   # 293 K exacto
geotherm_fn = fn.branching.conditional([
    (z >= GEO.nd(H_LAB), mantle_geotherm),             # >LAB
    (z >= GEO.nd(Hc),    lithosphere_geotherm),        # Hc–LAB
    (True,               crust_piecewise),             # 0–Hc
])


# In[42]:


Model.temperature.data[np.where((Model.materialField.evaluate(Model.mesh) == magma_chamber.index))] = GEO.nd((273 + cham_temp) * u.kelvin)


# In[43]:


#GEO.rcParams["advection.diffusion.method"] = "SUPG"


# ## Iniciar el modelo

# In[44]:


#Model.init_model(temperature=geotherm_fn, pressure= "lithostatic")
Model.init_model(temperature = "steady-state", pressure = "lithostatic")


# In[45]:


import numpy as np

# ============================================================
# 0) Máscara de cámara (1D) y coordenadas
# ============================================================
mat = Model.materialField.evaluate(Model.mesh).reshape(-1)   # 1D
mask_ch = (mat == magma_chamber.index)

coords = Model.mesh.data
x = coords[:, 0]
y = coords[:, 1]

# Nodos de cámara
ch_xy = coords[mask_ch, :2]   # (Nch, 2)

if ch_xy.shape[0] == 0:
    raise ValueError("mask_ch no tiene nodos. Revisá magma_chamber.index o la asignación de materiales.")

# ============================================================
# 1) Distancia a la cámara: distancia al nodo de cámara más cercano
# ============================================================
try:
    from scipy.spatial import cKDTree
    tree = cKDTree(ch_xy)
    dist, _ = tree.query(coords[:, :2], k=1)   # dist shape (N,)
except Exception:
    # Fallback (más lento): distancia mínima por chunks (sin scipy)
    # OJO: puede tardar si N es grande. Ajustá chunk si hace falta.
    dist = np.empty(coords.shape[0], dtype=float)
    chunk = 5000
    for i in range(0, coords.shape[0], chunk):
        xy = coords[i:i+chunk, :2]
        # (chunk, Nch) distancias: memoria puede ser grande si Nch es grande
        d2 = ((xy[:, None, 0] - ch_xy[None, :, 0])**2 +
              (xy[:, None, 1] - ch_xy[None, :, 1])**2)
        dist[i:i+chunk] = np.sqrt(d2.min(axis=1))

# ============================================================
# 2) Peso suave desde el borde de la cámara
#    delta controla el espesor del halo (misma unidad que coords)
# ============================================================
delta = GEO.nonDimensionalize(1 * u.km)  # si coords están en km -> 0.4 km; si están en m -> 400.0

# w=1 en dist=0, w~0 para dist >> delta
w = 0.5 * (1.0 - np.tanh(dist / delta))
w = np.clip(w, 0.0, 1.0).astype(float)

# ============================================================
# 3) Construir Tnew = mezcla suave entre Tc y Tbg
# ============================================================
Tc_K = GEO.nonDimensionalize((float(cham_temp) + 273.15) * u.kelvin) #float(cham_temp) + 273.15
#Tbg  = geotherm_fn.evaluate(Model.mesh).reshape(-1).astype(float)
Tbg = Model.temperature.data[:, 0].copy().astype(float)  # <- background steady-state


Tnew = w * Tc_K + (1.0 - w) * Tbg

# ============================================================
# 4) Asignación al campo de temperatura
#    Opción A: asignar en TODO el dominio (simple)
# ============================================================
Model.temperature.data[:, 0] = Tnew

# ------------------------------------------------------------
# Opción B (recomendada): asignar SOLO donde w > eps
# (para no tocar regiones lejanas / aire si no querés)
# ------------------------------------------------------------
# eps = 1e-6
# mask_aff = (w > eps)
# Model.temperature.data[mask_aff, 0] = Tnew[mask_aff]


# In[46]:


Model.temperature.data[np.where((Model.materialField.evaluate(Model.mesh) == magma_chamber.index))] = GEO.nd((273 + cham_temp) * u.kelvin)
Model.temperature.data[np.where((Model.materialField.evaluate(Model.mesh) == air.index))] = GEO.nd(Tsup)
Model.temperature.data[np.where((Model.materialField.evaluate(Model.mesh) == stickyAir.index))] = GEO.nd(Tsup)


# In[47]:


if figs == True:

    # ------------------------------------------------------------
    # 1. Extraer coordenadas en km
    # ------------------------------------------------------------
    coords = Model.mesh.data  # (N, 2) array (x, z) en metros
    x, z   = np.asarray(GEO.dimensionalise(coords[:,0],u.km)), np.asarray(GEO.dimensionalise(coords[:,1],u.km))

    # 2. Valores de temperatura,viscosidad, densidad
    T  = GEO.dimensionalise(Model.temperature, u.degC).evaluate(Model.mesh).flatten()
    η  = GEO.dimensionalise(Model.viscosityField, u.pascal*u.second).evaluate(Model.mesh).flatten()
    q  = GEO.dimensionalise(Model.densityField, u.kilogram / u.metre**3).evaluate(Model.mesh).flatten()
    mat= Model.materialField.evaluate(Model.mesh).flatten()   # <--- campo de materiales



    # ------------------------------------------------------------
    # 3. Rejilla regular en km
    # ------------------------------------------------------------
    nx, nz = 300, 150
    xi = np.linspace(x.min(), x.max(), nx)
    zi = np.linspace(z.min(), z.max(), nz)
    Xi, Zi = np.meshgrid(xi, zi)

    Ti = griddata((x, z), T, (Xi, Zi), method='linear')
    ηi = griddata((x, z), η, (Xi, Zi), method='linear')
    qi = griddata((x, z), q, (Xi, Zi), method='linear')
    mi  = griddata((x, z), mat,(Xi, Zi), method='nearest')  # usar nearest para categorías


    # ------------------------------------------------------------
    # 4. Ploteo
    # ------------------------------------------------------------

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6),
                                   sharex=True, constrained_layout=True)

    # Temperatura
    pcm1 = ax1.pcolormesh(Xi, Zi, Ti, shading='auto', cmap='coolwarm')
    ax1.set_title("Temperatura (C)")
    ax1.set_ylabel("Profundidad (km)")
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm1, ax=ax1, shrink=0.1, label="T (C)")

    # Viscosidad (log₁₀)
    pcm2 = ax2.pcolormesh(Xi, Zi, np.log10(ηi), shading='auto', cmap='Spectral_r')
    ax2.set_title(r"$\log_{10}(\eta)$  [Pa·s]")
    ax2.set_ylabel("Profundidad (km)")
    ax2.set_xlabel("Distancia horizontal (km)")
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm2, ax=ax2, shrink=0.1, label=r"log$_{10}(\eta)$")


    # Density
    pcm3 = ax3.pcolormesh(Xi, Zi, qi, shading='auto', cmap='plasma')
    ax3.set_title("Densidad (kg m$^{-3}$)")
    ax3.set_ylabel("Profundidad (km)")
    ax3.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm3, ax=ax3, shrink=0.1, label="ρ (kg m$^{-3}$)")

    # Material
    pcm4 = ax4.pcolormesh(Xi, Zi, mi, shading='auto', cmap='tab10')
    ax4.set_title("Material ID")
    ax4.set_ylabel("Profundidad (km)")
    ax4.set_xlabel("Distancia horizontal (km)")
    ax4.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm4, ax=ax4, shrink=0.1, label="Material")


    # Título global: tiempo del modelo
    fig.suptitle(f"Fields at t = {Model.time:.2f}", fontsize=14)



    plt.show()


# In[48]:


if figs == True:

    # --- Perfil de viscosidad en el centro del modelo (usando ηi) ---
    # índice de la columna más cercana al centro
    jmid = np.argmin(np.abs(xi - 0.5*(xi.min()+xi.max())))

    z_prof_km   = Zi[:, jmid]             # z (km), probablemente negativa hacia abajo
    eta_prof    = ηi[:, jmid]             # Pa·s
    depth_km    = -z_prof_km              # profundidad positiva

    # Plot (log10 de la viscosidad)
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(4,6))
    plt.plot(np.log10(eta_prof), depth_km)
    plt.gca().invert_yaxis()  # profundidad creciente hacia abajo en el eje y
    plt.xlabel(r'$\log_{10}(\eta)$ [Pa·s]')
    plt.ylabel('Profundidad (km)')
    plt.title('Perfil de viscosidad en el centro')
    plt.tight_layout()
    plt.show()

    # (opcional) Exportar a CSV
    import pandas as pd
    pd.DataFrame({'depth_km': depth_km, 'log10_eta': np.log10(eta_prof)}).to_csv(output_dir + '/perfil_viscosidad_centro.csv', index=False)


# In[49]:


if figs == True:

    # --- Perfil de temperatura en el centro del modelo (usando Ti) ---
    # índice de la columna más cercana al centro horizontal
    jmid = np.argmin(np.abs(xi - 0.5*(xi.min()+xi.max())))

    z_prof_km   = Zi[:, jmid]       # coordenada vertical (km, negativa hacia abajo)
    T_prof      = Ti[:, jmid]       # Temperatura (°C) interpolada
    depth_km    = -z_prof_km        # profundidad positiva

    # Plot
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(4,6))
    plt.plot(T_prof, depth_km)
    plt.gca().invert_yaxis()  # profundidad creciente hacia abajo
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Profundidad (km)')
    plt.title('Perfil de temperatura en el centro')
    plt.tight_layout()
    plt.show()

    # (opcional) Exportar a CSV
    import pandas as pd
    pd.DataFrame({'depth_km': depth_km, 'T_C': T_prof}).to_csv(output_dir + '/perfil_temperatura_centro.csv', index=False)


# In[50]:


GEO.rcParams["initial.nonlinear.tolerance"]=1e-3
#GEO.rcParams["advection.diffusion.method"] = "SLCN"
GEO.rcParams["advection.diffusion.method"] = "SUPG"


# In[51]:


Model.solver.set_inner_method("mumps")
Model.solver.set_penalty(1e6)
Model.outputDir = output_dir


# In[52]:


Total_Time = 1000000
Time_SUPG = 40000
ch_interval = Total_Time/100


# In[53]:


Model.run_for(Time_SUPG * u.years, checkpoint_interval = 5000 * u.years)
#Model.run_for(Total_Time * u.years, checkpoint_interval = ch_interval * u.years)



# In[ ]:


GEO.rcParams["advection.diffusion.method"] = "SLCN"
Rest_Time = Total_Time - Time_SUPG
Model.run_for(Rest_Time * u.years, checkpoint_interval = ch_interval * u.years, restartStep=-1)
#Model.run_for((ch_interval * 2), checkpoint_interval = ch_interval * u.years, restartStep=-1)


# In[55]:


if figs == True:

    import matplotlib.pyplot as plt

    # Definir niveles de isoterma (en °C)
    isotherms = [100, 200, 300, 400, 500, 600, 700, 900, 1000]

    fig, ax = plt.subplots(figsize=(5, 10))

    # Mapa de temperatura
    cmap = plt.get_cmap("coolwarm")
    temp_plot = ax.pcolormesh(Xi, Zi, Ti, shading='auto', cmap=cmap)
    cbar = fig.colorbar(temp_plot, ax=ax, shrink=0.5, label="Temperatura (°C)")

    # Contornos de isoterma
    CS = ax.contour(Xi, Zi, Ti, levels=isotherms, colors='cyan', linewidths=2)
    ax.clabel(CS, inline=True, fontsize=10, fmt='%d °C')

    # Zona de fusión parcial estimada (>800 °C por ejemplo)
    #ax.contourf(Xi, Zi, Ti, levels=[800, Ti.max()], colors='red', alpha=0.5)

    # Estética
    ax.set_title("Distribución térmica con líneas de isoterma")
    ax.set_xlabel("Distancia horizontal (km)")
    ax.set_ylabel("Profundidad (km)")
    ax.set_yticks([-20,-18,-16,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,4])
    #ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# In[56]:


if figs == True:

    # --- Perfil de viscosidad en el centro del modelo (usando ηi) ---
    # índice de la columna más cercana al centro
    jmid = np.argmin(np.abs(xi - 0.5*(xi.min()+xi.max())))

    z_prof_km   = Zi[:, jmid]             # z (km), probablemente negativa hacia abajo
    eta_prof    = ηi[:, jmid]             # Pa·s
    depth_km    = -z_prof_km              # profundidad positiva

    # Plot (log10 de la viscosidad)
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(4,6))
    plt.plot(np.log10(eta_prof), depth_km)
    plt.gca().invert_yaxis()  # profundidad creciente hacia abajo en el eje y
    plt.xlabel(r'$\log_{10}(\eta)$ [Pa·s]')
    plt.ylabel('Profundidad (km)')
    plt.title('Perfil de viscosidad en el centro')
    plt.tight_layout()
    plt.show()

    # (opcional) Exportar a CSV
    import pandas as pd
    pd.DataFrame({'depth_km': depth_km, 'log10_eta': np.log10(eta_prof)}).to_csv('perfil_viscosidad_centro.csv', index=False)


# In[57]:


if figs == True:

    # ------------------------------------------------------------
    # 1. Extraer coordenadas en km
    # ------------------------------------------------------------
    coords = Model.mesh.data  # (N, 2) array (x, z) en metros
    x, z   = np.asarray(GEO.dimensionalise(coords[:,0],u.km)), np.asarray(GEO.dimensionalise(coords[:,1],u.km))

    # 2. Valores de temperatura,viscosidad, densidad
    T  = GEO.dimensionalise(Model.temperature, u.degC).evaluate(Model.mesh).flatten()
    η  = GEO.dimensionalise(Model.viscosityField, u.pascal*u.second).evaluate(Model.mesh).flatten()
    q  = GEO.dimensionalise(Model.densityField, u.kilogram / u.metre**3).evaluate(Model.mesh).flatten()
    mat= Model.materialField.evaluate(Model.mesh).flatten()   # <--- campo de materiales



    # ------------------------------------------------------------
    # 3. Rejilla regular en km
    # ------------------------------------------------------------
    nx, nz = 300, 150
    xi = np.linspace(x.min(), x.max(), nx)
    zi = np.linspace(z.min(), z.max(), nz)
    Xi, Zi = np.meshgrid(xi, zi)

    Ti = griddata((x, z), T, (Xi, Zi), method='linear')
    ηi = griddata((x, z), η, (Xi, Zi), method='linear')
    qi = griddata((x, z), q, (Xi, Zi), method='linear')
    mi  = griddata((x, z), mat,(Xi, Zi), method='nearest')  # usar nearest para categorías


    # ------------------------------------------------------------
    # 4. Ploteo
    # ------------------------------------------------------------

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 7),
                                   sharex=True, constrained_layout=True)

    # Temperatura
    pcm1 = ax1.pcolormesh(Xi, Zi, Ti, shading='auto', cmap='coolwarm')
    ax1.set_title("Temperatura (C)")
    ax1.set_ylabel("Profundidad (km)")
    fig.colorbar(pcm1, ax=ax1, shrink=0.7, label="T (C)")

    # Viscosidad (log₁₀)
    pcm2 = ax2.pcolormesh(Xi, Zi, np.log10(ηi), shading='auto', cmap='Spectral_r')
    ax2.set_title(r"$\log_{10}(\eta)$  [Pa·s]")
    ax2.set_ylabel("Profundidad (km)")
    ax2.set_xlabel("Distancia horizontal (km)")
    fig.colorbar(pcm2, ax=ax2, shrink=0.7, label=r"log$_{10}(\eta)$")


    # Density
    pcm3 = ax3.pcolormesh(Xi, Zi, qi, shading='auto', cmap='plasma')
    ax3.set_title("Densidad (kg m$^{-3}$)")
    ax3.set_ylabel("Profundidad (km)")
    fig.colorbar(pcm3, ax=ax3, shrink=0.7, label="ρ (kg m$^{-3}$)")

    # Material
    pcm4 = ax4.pcolormesh(Xi, Zi, mi, shading='auto', cmap='tab20')
    ax4.set_title("Material ID")
    ax4.set_ylabel("Profundidad (km)")
    ax4.set_xlabel("Distancia horizontal (km)")
    fig.colorbar(pcm4, ax=ax4, shrink=0.7, label="Material")


    # Título global: tiempo del modelo
    fig.suptitle(f"Fields at t = {Model.time:.2f}", fontsize=14)



    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




