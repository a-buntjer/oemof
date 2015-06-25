# -*- coding: utf-8 -*-

from time import time
from optimization_model import *

import components as cp
import random

timesteps = [t for t in range(10)]

# Busses (1 Coal, 2 Elec, 1 Thermal)
bus_coal = cp.Bus(uid="coal_bus", type="coal")
bus_el1 = cp.Bus(uid="region_1", type="elec")
bus_el2 = cp.Bus(uid="region_2", type="elec")
bus_th1 = cp.Bus(uid="district_heating", type="th")

# Renewable sources
wind_r1 = cp.Source(uid="wind_r1", outputs=[bus_el1],
                    val=[random.gauss(15,1) for i in timesteps])
wind_r2 = cp.Source(uid="wind_r2", outputs=[bus_el2],
                    val=[random.gauss(10,1) for i in timesteps])
solar = cp.Source(uid="solar_heat", outputs=[bus_th1],
                  val=[random.gauss(3,1) for i in timesteps])

# Commodity sources
r_coal = cp.Source(uid="r_coal", outputs=[bus_coal],
                   val=[1000 for t in timesteps])

# Sinks
demand_r1 = cp.Sink(uid="demand_r1", inputs=[bus_el1],
                    val=[random.gauss(30,4) for i in timesteps])
demand_r2 = cp.Sink(uid="demand_r2", inputs=[bus_el2],
                    val=[random.gauss(50,4) for i in timesteps])
demand_th = cp.Sink(uid="demand_th", inputs=[bus_th1],
                    val=[random.gauss(50,4) for i in timesteps])

# Simple Transformer for region_1
SF_region_1 = cp.SimpleTransformer(uid='SFr1', inputs=[bus_coal],
                                   outputs=[bus_el2], in_max=200, out_max=100,
                                   eta=0.5, opex_var=10)
# Simple Transformer for region_2
SF_region_2 = cp.SimpleTransformer(uid='SFr2', inputs=[bus_coal],
                                   outputs=[bus_el1], in_max=200, out_max=100,
                                   eta=0.4, opex_var=20)
# Boiler for district heating
SF_district_heating = cp.SimpleTransformer(uid='Boiler', inputs=[bus_coal],
                                           outputs=[bus_th1], in_max=200,
                                           eta=0.9, opex_var=40)
# Storage electric for region_1
SS_region_1 = cp.SimpleStorage(uid="Storage", outputs=[bus_el1],
                               inputs=[bus_th1],
                               opt_param={'soc_max':10, 'soc_min':1})

# group all components
buses = [bus_coal, bus_el1, bus_el2, bus_th1]
components = [wind_r1, wind_r2, solar, r_coal] + [demand_r1, demand_r2] + \
              [SF_region_1] + [SF_region_2] + [SF_district_heating] + [SS_region_1]
#I, O = io_sets(objs_schp)


t0 = time()

# create optimization model
om = opt_model(buses, components, timesteps=timesteps, invest=True)

t1 = time()
building_time = t1 - t0
print('Building time', building_time)
# create model instance

print('Creating instance...')
instance = om.create(report_timing=True)
#instance.pprint()

print('Solving model...')
t0 = time()
instance = solve_opt_model(instance=instance, solver='gurobi',
                           options={'stream':True}, debug=True)
t1 = time()
solving_time = t1 - t0
print('Solving time:', solving_time)

#instance.source_val['s1'] = [random.gauss(-10,4) for i in timesteps]
#instance.del_component('w')
#instance.w = po.Var(instance.edges, instance.timesteps, bounds=w_max_rule,
#                    within=po.NonNegativeReals)

  #for idx in instance.w:
  #  print('Edge:'+str(idx), ', weight:'+str(instance.w[idx].value))