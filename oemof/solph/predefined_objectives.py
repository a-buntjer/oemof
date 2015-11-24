# -*- coding: utf-8 -*-
"""
This module contains predefined objectives that can be used to model
energy systems.

@author: Simon Hilpert (simon.hilpert@fh-flensburg.de)
"""
try:
    import objective_expressions as objexpr
except:
    from . import objective_expressions as objexpr

import pyomo.environ as po
import oemof.solph as solph

from ..core.network.entities import Bus
from ..core.network.entities.components import transformers as transformer
from ..core.network.entities.components import sources as source

def minimize_cost(self, c_blocks=(), r_blocks=()):
    """ Builds objective function that minimises the total costs.

    Costs included are:
                        opex_var,
                        opex_fix,
                        curtailment_costs (dispatch sources),
                        annualised capex (investment components)

    Parameters:
    ------------
    self : pyomo model instance
    c_blocks : pyomo blocks containing components that are included in
               cost terms of objective function
    r_blocks : pyomo blocks containing components that are included in revenue
               terms of objective function
    """
    expr = 0
    c_blocks = (str(transformer.Simple), str(transformer.CHP),
                str(source.FixedSource))
    r_blocks = ()

    blocks = [block for block in self.block_data_objects(active=True)
              if not isinstance(block,
                                solph.optimization_model.OptimizationModel)]

    for block in blocks:
        if block.name in c_blocks:
            if block.name == 'simple_storage':
                ref = 'capacity'
                expr += objexpr.add_opex_var(self, self.simple_storage,
                                             ref='input')
            else:
                ref = 'output'
            # variable costs

            expr += objexpr.add_opex_var(self, block, ref='output')
            # fix costs
            expr += objexpr.add_opex_fix(self, block, ref=ref)
            # investment costs
            if block.optimization_options.get('investment', False):
                expr += objexpr.add_capex(self, block, ref=ref)
            # revenues
        if block.name in r_blocks:
            expr += objexpr.add_revenues(self, block, ref='output')

    # costs for dispatchable sources
    if hasattr(self, 'dispatch_source'):
        expr += objexpr.add_curtailment_costs(self, self.dispatch_source)

    if getattr(self, str(Bus)).shortage_uids:
        expr += objexpr.add_shortage_slack_costs(self, block)
    # artificial costs for excess or shortage
    if getattr(self, str(Bus)).excess_uids:
        expr += objexpr.add_excess_slack_costs(self, block)

    self.objective = po.Objective(expr=expr)
