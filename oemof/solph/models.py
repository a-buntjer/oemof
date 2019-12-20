# -*- coding: utf-8 -*-
"""Solph Optimization Models

This file is part of project oemof (github.com/oemof/oemof). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location oemof/oemof/solph/models.py

SPDX-License-Identifier: GPL-3.0-or-later
"""

import pyomo.environ as po
from pyomo.opt import SolverFactory
from pyomo.core.plugins.transform.relax_integrality import RelaxIntegrality
from oemof.solph import blocks
from oemof.solph.plumbing import sequence
from oemof.outputlib import processing
import warnings
import logging


class BaseModel(po.ConcreteModel):
    """ The BaseModel for other solph-models (Model, MultiPeriodModel, etc.)

    Parameters
    ----------
    energysystem : EnergySystem object
        Object that holds the nodes of an oemof energy system graph
    constraint_groups : list (optional)
        Solph looks for these groups in the given energy system and uses them
        to create the constraints of the optimization problem.
        Defaults to :const:`Model.CONSTRAINTS`
    objective_weighting : array like (optional)
        Weights used for temporal objective function
        expressions. If nothing is passed `timeincrement` will be used which
        is calculated from the freq length of the energy system timeindex .
    auto_construct : boolean
        If this value is true, the set, variables, constraints, etc. are added,
        automatically when instantiating the model. For sequential model
        building process set this value to False
        and use methods `_add_parent_block_sets`,
        `_add_parent_block_variables`, `_add_blocks`, `_add_objective`

    Attributes:
    -----------
    timeincrement : sequence
        Time increments.
    flows : list
        Flows of the model.
    name : str
        Name of the model.
    es : solph.EnergySystem
        Energy system of the model.
    meta : pyomo.opt.results.results_.SolverResults or None
        Solver results.
    dual : ... or None
    rc : ... or None

    """
    CONSTRAINT_GROUPS = []

    def __init__(self, energysystem, **kwargs):
        super().__init__()

        # ########################  Arguments #################################

        self.name = kwargs.get('name', type(self).__name__)
        self.es = energysystem
        self.timeincrement = sequence(kwargs.get('timeincrement', None))

        if self.timeincrement[0] is None:
            try:
                self.timeincrement = sequence(
                    self.es.timeindex.freq.nanos / 3.6e12)
            except AttributeError:
                msg = ("No valid time increment found. Please pass a valid "
                       "timeincremet parameter or pass an EnergySystem with "
                       "a valid time index. Please note that a valid time"
                       "index need to have a 'freq' attribute.")
                raise AttributeError(msg)

        self.objective_weighting = kwargs.get('objective_weighting',
                                              self.timeincrement)

        self._constraint_groups = (type(self).CONSTRAINT_GROUPS +
                                   kwargs.get('constraint_groups', []))

        self._constraint_groups += [i for i in self.es.groups
                                    if hasattr(i, 'CONSTRAINT_GROUP') and
                                    i not in self._constraint_groups]

        self.flows = self.es.flows()

        self.solver_results = None
        self.dual = None
        self.rc = None
        if kwargs.get("auto_construct", True):
            self._construct()

    def _construct(self):
        """
        """
        self._add_parent_block_sets()
        self._add_parent_block_variables()
        self._add_child_blocks()
        self._add_objective()

    def _add_parent_block_sets(self):
        """" Method to create all sets located at the parent block, i.e. the
        model itself as they are to be shared across all model components.
        """
        pass

    def _add_parent_block_variables(self):
        """" Method to create all variables located at the parent block,
        i.e. the model itself as these variables  are to be shared across
        all model components.
        """
        pass

    def _add_child_blocks(self):
        """ Method to add the defined child blocks for components that have
        been grouped in the defined constraint groups.
        """

        for group in self._constraint_groups:
            # create instance for block
            block = group()
            # Add block to model
            self.add_component(str(block), block)
            # create constraints etc. related with block for all nodes
            # in the group
            block._create(group=self.es.groups.get(group))

    def _add_objective(self, sense=po.minimize, update=False):
        """ Method to sum up all objective expressions from the child blocks
        that have been created. This method looks for `_objective_expression`
        attribute in the block definition and will call this method to add
        their return value to the objective function.
        """
        if update:
            self.del_component('objective')

        expr = 0

        for block in self.component_data_objects():
            if hasattr(block, '_objective_expression'):
                expr += block._objective_expression()

        self.objective = po.Objective(sense=sense, expr=expr)

    def receive_duals(self):
        """ Method sets solver suffix to extract information about dual
        variables from solver. Shadow prices (duals) and reduced costs (rc) are
        set as attributes of the model.

        """
        # shadow prices
        self.dual = po.Suffix(direction=po.Suffix.IMPORT)
        # reduced costs
        self.rc = po.Suffix(direction=po.Suffix.IMPORT)

    def results(self):
        """ Returns a nested dictionary of the results of this optimization
        """
        return processing.results(self)

    def solve(self, solver='cbc', solver_io='lp', **kwargs):
        r""" Takes care of communication with solver to solve the model.

        Parameters
        ----------
        solver : string
            solver to be used e.g. "glpk","gurobi","cplex"
        solver_io : string
            pyomo solver interface file format: "lp","python","nl", etc.
        \**kwargs : keyword arguments
            Possible keys can be set see below:

        Other Parameters
        ----------------
        solve_kwargs : dict
            Other arguments for the pyomo.opt.SolverFactory.solve() method
            Example : {"tee":True}
        cmdline_options : dict
            Dictionary with command line options for solver e.g.
            {"mipgap":"0.01"} results in "--mipgap 0.01"
            {"interior":" "} results in "--interior"
            Gurobi solver takes numeric parameter values such as
            {"method": 2}

        """
        solve_kwargs = kwargs.get('solve_kwargs', {})
        solver_cmdline_options = kwargs.get("cmdline_options", {})

        opt = SolverFactory(solver, solver_io=solver_io)
        # set command line options
        options = opt.options
        for k in solver_cmdline_options:
            options[k] = solver_cmdline_options[k]

        solver_results = opt.solve(self, **solve_kwargs)

        status = solver_results["Solver"][0]["Status"].key
        termination_condition = (
            solver_results["Solver"][0]["Termination condition"].key)

        if status == "ok" and termination_condition == "optimal":
            logging.info("Optimization successful...")
            self.es.results = solver_results
            self.solver_results = solver_results
        else:
            msg = ("Optimization ended with status {0} and termination "
                   "condition {1}")
            warnings.warn(msg.format(status, termination_condition),
                          UserWarning)
            self.es.results = solver_results
            self.solver_results = solver_results

        return solver_results

    def relax_problem(self):
        """Relaxes integer variables to reals of optimization model self."""
        relaxer = RelaxIntegrality()
        relaxer._apply_to(self)

        return self


class Model(BaseModel):
    """ An  energy system model for operational and investment
    optimization.

    Parameters
    ----------
    energysystem : EnergySystem object
        Object that holds the nodes of an oemof energy system graph
    constraint_groups : list
        Solph looks for these groups in the given energy system and uses them
        to create the constraints of the optimization problem.
        Defaults to :const:`Model.CONSTRAINTS`

    **The following basic sets are created**:

    NODES :
        A set with all nodes of the given energy system.

    TIMESTEPS :
        A set with all timesteps of the given time horizon.

    FLOWS :
        A 2 dimensional set with all flows. Index: `(source, target)`

    **The following basic variables are created**:

    flow
        Flow from source to target indexed by FLOWS, TIMESTEPS.
        Note: Bounds of this variable are set depending on attributes of
        the corresponding flow object.

    """
    CONSTRAINT_GROUPS = [blocks.Bus, blocks.Transformer,
                         blocks.InvestmentFlow, blocks.Flow,
                         blocks.NonConvexFlow]

    def __init__(self, energysystem, **kwargs):
        super().__init__(energysystem, **kwargs)

    def _add_parent_block_sets(self):
        """
        """
        # set with all nodes
        self.NODES = po.Set(initialize=[n for n in self.es.nodes])

        # pyomo set for timesteps of optimization problem
        self.TIMESTEPS = po.Set(initialize=range(len(self.es.timeindex)),
                                ordered=True)

        # previous timesteps
        previous_timesteps = [x - 1 for x in self.TIMESTEPS]
        previous_timesteps[0] = self.TIMESTEPS.last()

        self.previous_timesteps = dict(zip(self.TIMESTEPS, previous_timesteps))

        # pyomo set for all flows in the energy system graph
        self.FLOWS = po.Set(initialize=self.flows.keys(),
                            ordered=True, dimen=2)

        self.BIDIRECTIONAL_FLOWS = po.Set(initialize=[
            k for (k, v) in self.flows.items() if hasattr(v, 'bidirectional')],
                                          ordered=True, dimen=2,
                                          within=self.FLOWS)

        self.UNIDIRECTIONAL_FLOWS = po.Set(
            initialize=[k for (k, v) in self.flows.items() if not
                        hasattr(v, 'bidirectional')],
            ordered=True, dimen=2, within=self.FLOWS)

    def _add_parent_block_variables(self):
        """
        """
        self.flow = po.Var(self.FLOWS, self.TIMESTEPS,
                           within=po.Reals)

        for (o, i) in self.FLOWS:
            for t in self.TIMESTEPS:
                if (o, i) in self.UNIDIRECTIONAL_FLOWS:
                    self.flow[o, i, t].setlb(0)
                if self.flows[o, i].nominal_value is not None:
                    self.flow[o, i, t].setub(self.flows[o, i].max[t] *
                                             self.flows[o, i].nominal_value)

                    if (self.flows[o, i].actual_value[t] is not None):
                        # pre- optimized value of flow variable
                        self.flow[o, i, t].value = (
                            self.flows[o, i].actual_value[t] *
                            self.flows[o, i].nominal_value)
                        # fix variable if flow is fixed
                        if self.flows[o, i].fixed:
                            self.flow[o, i, t].fix()
                    if not (self.flows[o, i].nonconvex):
                        # lower bound of flow variable
                        self.flow[o, i, t].setlb(
                            self.flows[o, i].min[t] *
                            self.flows[o, i].nominal_value)


class MultiPeriodModel(BaseModel):
    """ An  multi period energy system model for operational and investment
    optimization with a rolling horizon.

    Parameters
    ----------
    energysystem : EnergySystem object
        Object that holds the nodes of an oemof energy system graph
    constraint_groups : list
        Solph looks for these groups in the given energy system and uses them
        to create the constraints of the optimization problem.
        Defaults to :const:`Model.CONSTRAINTS`

    **The following basic sets are created**:

    NODES :
        A set with all nodes of the given energy system.

    TIMESTEPS :
        A set with all timesteps of the given time horizon.

    FLOWS :
        A 2 dimensional set with all flows. Index: `(source, target)`

    **The following basic variables are created**:

    flow
        Flow from source to target indexed by FLOWS, TIMESTEPS.
        Note: Bounds of this variable are set depending on attributes of
        the corresponding flow object.

    """
    CONSTRAINT_GROUPS = [blocks.Bus, blocks.Transformer,
                         blocks.InvestmentFlow, blocks.Flow,
                         blocks.NonConvexFlow, blocks.RollingHorizonFlow]

    def __init__(self, energysystem, interval_length, period, **kwargs):
        self.period = period
        self.interval_length = interval_length
        self.last_t_in_interval = interval_length - 1
        self.multiperiod_results = None
        self.multi_period_timeindex = energysystem.timeindex
        energysystem.timeindex =\
            self.multi_period_timeindex[:self.interval_length]
        super().__init__(energysystem, **kwargs)

    @property
    def last_t_in_interval(self):
        _last_t_in_interval = self.interval_length - 1
        return _last_t_in_interval

    @last_t_in_interval.setter
    def last_t_in_interval(self, interval_length):
        if self.interval_length < self.period:
            raise ValueError(
                    "interval_length must be greater or equal to period")

    @property
    def total_optimization_period(self):
        _total_optimization_period = [x for x in range(0,
                                      len(self.multi_period_timeindex),
                                      self.period)]
        return _total_optimization_period

    def _add_parent_block_sets(self):
        """
        """
        # set with all nodes
        self.NODES = po.Set(initialize=[n for n in self.es.nodes])

        # pyomo set for timesteps of optimization problem
        self.TIMESTEPS = po.Set(initialize=range(len(self.es.timeindex)),
                                ordered=True)

        # previous timesteps
        previous_timesteps = [x - 1 for x in self.TIMESTEPS]
        previous_timesteps[0] = self.TIMESTEPS.last()

        self.previous_timesteps = dict(zip(self.TIMESTEPS, previous_timesteps))

        # pyomo set for all flows in the energy system graph
        self.FLOWS = po.Set(initialize=self.flows.keys(),
                            ordered=True, dimen=2)

        self.BIDIRECTIONAL_FLOWS = po.Set(initialize=[
            k for (k, v) in self.flows.items() if hasattr(v, 'bidirectional')],
                                          ordered=True, dimen=2,
                                          within=self.FLOWS)

        self.UNIDIRECTIONAL_FLOWS = po.Set(
            initialize=[k for (k, v) in self.flows.items() if not
                        hasattr(v, 'bidirectional')],
            ordered=True, dimen=2, within=self.FLOWS)

    def _add_parent_block_variables(self):
        """
        """
        self.flow = po.Var(self.FLOWS, self.TIMESTEPS,
                           within=po.Reals)

        for (o, i) in self.FLOWS:
            for t in self.TIMESTEPS:
                if (o, i) in self.UNIDIRECTIONAL_FLOWS:
                    self.flow[o, i, t].setlb(0)
                if self.flows[o, i].nominal_value is not None:
                    self.flow[o, i, t].setub(self.flows[o, i].max[t] *
                                             self.flows[o, i].nominal_value)

                    if (self.flows[o, i].actual_value[t] is not None and
                            self.flows[o, i].rollinghorizon is None):
                        # pre- optimized value of flow variable
                        self.flow[o, i, t].value = (
                            self.flows[o, i].actual_value[t] *
                            self.flows[o, i].nominal_value)
                        # fix variable if flow is fixed
                        if self.flows[o, i].fixed:
                            self.flow[o, i, t].fix()
                    if not (self.flows[o, i].nonconvex or
                            self.flows[o, i].rollinghorizon):
                        # lower bound of flow variable
                        self.flow[o, i, t].setlb(
                            self.flows[o, i].min[t] *
                            self.flows[o, i].nominal_value)
            if self.flows[o, i].rollinghorizon:
                self.flows[o, i].rollinghorizon.t_last =\
                    self.last_t_in_interval
                self.flows[o, i].rollinghorizon.setup_multi_period_containers(
                        len(self.multi_period_timeindex))
                if self.flows[o, i].nominal_value is not None:
                    if self.flows[o, i].rollinghorizon.ramp_limit_down\
                            is not None:
                        self.flows[o, i].rollinghorizon.R_down =\
                            self.flows[o, i].rollinghorizon.ramp_limit_down *\
                            self.flows[o, i].nominal_value *\
                            self.flows[o, i].rollinghorizon.tau
                    if self.flows[o, i].rollinghorizon.ramp_limit_up\
                            is not None:
                        self.flows[o, i].rollinghorizon.R_up =\
                            self.flows[o, i].rollinghorizon.ramp_limit_up *\
                            self.flows[o, i].nominal_value *\
                            self.flows[o, i].rollinghorizon.tau
                    if self.flows[o, i].rollinghorizon.ramp_limit_start_up\
                            is not None:
                        self.flows[o, i].rollinghorizon.R_start =\
                            self.flows[o, i].rollinghorizon.ramp_limit_start_up *\
                            self.flows[o, i].nominal_value *\
                            self.flows[o, i].rollinghorizon.tau
                    if self.flows[o, i].rollinghorizon.ramp_limit_shut_down\
                            is not None:
                        self.flows[o, i].rollinghorizon.R_shutdown =\
                            self.flows[o, i].rollinghorizon.ramp_limit_shut_down *\
                            self.flows[o, i].nominal_value *\
                            self.flows[o, i].rollinghorizon.tau

    def solve(self, solver='cbc', solver_io='lp', **kwargs):
        r""" Takes care of communication with solver to solve the model.

        Parameters
        ----------
        solver : string
            solver to be used e.g. "glpk","gurobi","cplex"
        solver_io : string
            pyomo solver interface file format: "lp","python","nl", etc.
        \**kwargs : keyword arguments
            Possible keys can be set see below:

        Other Parameters
        ----------------
        solve_kwargs : dict
            Other arguments for the pyomo.opt.SolverFactory.solve() method
            Example : {"tee":True}
        cmdline_options : dict
            Dictionary with command line options for solver e.g.
            {"mipgap":"0.01"} results in "--mipgap 0.01"
            {"interior":" "} results in "--interior"
            Gurobi solver takes numeric parameter values such as
            {"method": 2}

        """
        solve_kwargs = kwargs.get('solve_kwargs', {})
        solver_cmdline_options = kwargs.get("cmdline_options", {})

        opt = SolverFactory(solver, solver_io=solver_io)
        # set command line options
        options = opt.options
        for k in solver_cmdline_options:
            options[k] = solver_cmdline_options[k]
        self.es.results = {}
        self.solver_results = {}
        for t_first in self.total_optimization_period:
            self.es.timeindex = self.multi_period_timeindex[
                    t_first:t_first+self.interval_length]
            solver_results = opt.solve(self, **solve_kwargs)

            status = solver_results["Solver"][0]["Status"].key
            termination_condition = (
                solver_results["Solver"][0]["Termination condition"].key)

            if status == "ok" and termination_condition == "optimal":
                logging.info(
                        f"Optimization of time interval: {t_first} successful...")
            else:
                msg = (f"Optimization ended in time interval: {t_first}"
                       "with status {0} and termination condition {1}")
                warnings.warn(msg.format(status, termination_condition),
                              UserWarning)
                raise SystemExit('Found no solution program ended!')
            self.es.results[t_first] = solver_results
            self.solver_results[t_first] = solver_results
            actual_loop_results = self.results()
            if self.multiperiod_results is None:
                self.multiperiod_results = actual_loop_results
            else:
                for key, value in actual_loop_results.items():
                    for key2, value2 in value.items():
                        self.multiperiod_results[key][key2] =\
                            value2.combine_first(
                                self.multiperiod_results[key][key2])
            # Set result values for next loop
            t_first_next = t_first + self.period
            if t_first_next + self.TIMESTEPS[-1] >\
                    (len(self.multi_period_timeindex)):
                break
            for (o, i) in self.FLOWS:
                # Set values for new loop
                if self.flows[o, i].rollinghorizon:
                    self.flows[o, i].rollinghorizon.t_first = t_first_next
                for t in self.TIMESTEPS:
                    if (self.flows[o, i].variable_costs[t] is not None):
                        try:
                            self.flows[o, i].variable_costs[t] =\
                                self.flows[o, i].variable_costs[t_first_next+t]
                        except IndexError:
                            pass
                    if self.flows[o, i].nominal_value is not None:
                        self.flow[o, i, t].setub(
                            self.flows[o, i].max[t_first_next+t] *
                            self.flows[o, i].nominal_value)
                        if (self.flows[o, i].actual_value[t] is not None):
                            self.flow[o, i, t].value = (
                                self.flows[o, i].actual_value[
                                        t_first_next+t] *
                                self.flows[o, i].nominal_value)
                            if self.flows[o, i].fixed:
                                self.flow[o, i, t].fix()
                        if not (self.flows[o, i].nonconvex or
                                self.flows[o, i].rollinghorizon):
                            # lower bound of flow variable
                            self.flow[o, i, t].setlb(
                                self.flows[o, i].min[t+t_first_next] *
                                self.flows[o, i].nominal_value)
                    if self.flows[o, i].rollinghorizon:
                        try:
                            self.flows[o, i].rollinghorizon.cold_start_costs[t] =\
                                self.flows[o, i].rollinghorizon.cold_start_costs[
                                    t_first_next+t]
                            self.flows[o, i].rollinghorizon.warm_start_costs =\
                                self.flows[o, i].rollinghorizon.warm_start_costs[
                                    t_first_next+t]
                            self.flows[o, i].rollinghorizon.hot_start_costs =\
                                self.flows[o, i].rollinghorizon.hot_start_costs[
                                    t_first_next+t]
                        except IndexError:
                            pass
                        except TypeError:
                            pass
                    # Set values for new loop based on old timestep
                    if self.flows[o, i].rollinghorizon and\
                            self.flows[o, i].nominal_value is not None:
                        self.flows[o, i].rollinghorizon.initial_min_flow =\
                            self.flows[o, i].min[t_first+self.period-1] *\
                            self.flows[o, i].nominal_value
                    if self.flows[o, i].rollinghorizon:
                        self.flows[o, i].rollinghorizon.optimized_status[
                                t_first:t_first+self.interval_length] =\
                            list(self.RollingHorizonFlow.status[o, i, :]())
                        self.flows[o, i].rollinghorizon.optimized_flow[
                                t_first:t_first+self.interval_length] =\
                            list(self.flow[o, i, :]())
            self._add_objective(update=True)
            if hasattr(self, 'GenericStorageBlock'):
                for n in self.GenericStorageBlock.STORAGES:
                    self.GenericStorageBlock.init_cap[n].fix(
                        self.GenericStorageBlock.capacity[
                                n, self.period-1].value)
        return solver_results
