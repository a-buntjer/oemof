# -*- coding: utf-8 -*-

"""This module is designed to hold custom components with their classes and
associated individual constraints (blocks) and groupings. Therefore this
module holds the class definition and the block directly located by each other.

This file is part of project oemof (github.com/oemof/oemof). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location oemof/oemof/solph/custom.py

SPDX-License-Identifier: GPL-3.0-or-later
"""

from pyomo.core.base.block import SimpleBlock
from pyomo.environ import (Binary, Set, NonNegativeReals, Var, Constraint,
                           Expression, BuildAction)
import logging
import pandas as pd
import numpy as np
from oemof import network
from oemof.solph import components
from oemof.solph.network import Bus, Transformer, Flow
from oemof.solph.plumbing import sequence


class GenericCHP(network.Transformer):
    r"""
    Component `GenericCHP` to model combined heat and power plants.

    Can be used to model (combined cycle) extraction or back-pressure turbines
    and used a mixed-integer linear formulation. Thus, it induces more
    computational effort than the `ExtractionTurbineCHP` for the
    benefit of higher accuracy.

    The full set of equations is described in:
    Mollenhauer, E., Christidis, A. & Tsatsaronis, G.
    Evaluation of an energy- and exergy-based generic modeling
    approach of combined heat and power plants
    Int J Energy Environ Eng (2016) 7: 167.
    https://doi.org/10.1007/s40095-016-0204-6

    For a general understanding of (MI)LP CHP representation, see:
    Fabricio I. Salgado, P.
    Short - Term Operation Planning on Cogeneration Systems: A Survey
    Electric Power Systems Research (2007)
    Electric Power Systems Research
    Volume 78, Issue 5, May 2008, Pages 835-848
    https://doi.org/10.1016/j.epsr.2007.06.001

    Note
    ----
    An adaption for the flow parameter `H_L_FG_share_max` has been made to
    set the flue gas losses at maximum heat extraction `H_L_FG_max` as share of
    the fuel flow `H_F` e.g. for combined cycle extraction turbines.
    The flow parameter `H_L_FG_share_min` can be used to set the flue gas
    losses at minimum heat extraction `H_L_FG_min` as share of
    the fuel flow `H_F` e.g. for motoric CHPs.
    The boolean component parameter `back_pressure` can be set to model
    back-pressure characteristics.

    Also have a look at the examples on how to use it.

    Parameters
    ----------
    fuel_input : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the fuel input.
    electrical_output : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the electrical output. Related parameters like `P_max_woDH` are
        passed as attributes of the `oemof.Flow` object.
    heat_output : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the heat output. Related parameters like `Q_CW_min` are passed as
        attributes of the `oemof.Flow` object.
    Beta : list of numerical values
        Beta values in same dimension as all other parameters (length of
        optimization period).
    back_pressure : boolean
        Flag to use back-pressure characteristics. Set to `True` and
        `Q_CW_min` to zero for back-pressure turbines. See paper above for more
        information.

    Note
    ----
    The following sets, variables, constraints and objective parts are created
     * :py:class:`~oemof.solph.components.GenericCHPBlock`

    Examples
    --------
    >>> from oemof import solph
    >>> bel = solph.Bus(label='electricityBus')
    >>> bth = solph.Bus(label='heatBus')
    >>> bgas = solph.Bus(label='commodityBus')
    >>> ccet = solph.components.GenericCHP(
    ...    label='combined_cycle_extraction_turbine',
    ...    fuel_input={bgas: solph.Flow(
    ...        H_L_FG_share_max=[0.183])},
    ...    electrical_output={bel: solph.Flow(
    ...        P_max_woDH=[155.946],
    ...        P_min_woDH=[68.787],
    ...        Eta_el_max_woDH=[0.525],
    ...        Eta_el_min_woDH=[0.444])},
    ...    heat_output={bth: solph.Flow(
    ...        Q_CW_min=[10.552])},
    ...    Beta=[0.122], back_pressure=False)
    >>> type(ccet)
    <class 'oemof.solph.components.GenericCHP'>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fuel_input = kwargs.get('fuel_input')
        self.electrical_output = kwargs.get('electrical_output')
        self.heat_output = kwargs.get('heat_output')
        self.Beta = sequence(kwargs.get('Beta'))
        self.back_pressure = kwargs.get('back_pressure')
        self._alphas = None

        # map specific flows to standard API
        fuel_bus = list(self.fuel_input.keys())[0]
        fuel_flow = list(self.fuel_input.values())[0]
        fuel_bus.outputs.update({self: fuel_flow})

        self.outputs.update(kwargs.get('electrical_output'))
        self.outputs.update(kwargs.get('heat_output'))

    def _calculate_alphas(self):
        """
        Calculate alpha coefficients.

        A system of linear equations is created from passed capacities and
        efficiencies and solved to calculate both coefficients.
        """
        alphas = [[], []]

        eb = list(self.electrical_output.keys())[0]

        attrs = [self.electrical_output[eb].P_min_woDH,
                 self.electrical_output[eb].Eta_el_min_woDH,
                 self.electrical_output[eb].P_max_woDH,
                 self.electrical_output[eb].Eta_el_max_woDH]

        length = [len(a) for a in attrs if not isinstance(a, (int, float))]
        max_length = max(length)

        if all(len(a) == max_length for a in attrs):
            if max_length == 0:
                max_length += 1  # increment dimension for scalars from 0 to 1
            for i in range(0, max_length):
                A = np.array([[1, self.electrical_output[eb].P_min_woDH[i]],
                              [1, self.electrical_output[eb].P_max_woDH[i]]])
                b = np.array([self.electrical_output[eb].P_min_woDH[i] /
                              self.electrical_output[eb].Eta_el_min_woDH[i],
                              self.electrical_output[eb].P_max_woDH[i] /
                              self.electrical_output[eb].Eta_el_max_woDH[i]])
                x = np.linalg.solve(A, b)
                alphas[0].append(x[0])
                alphas[1].append(x[1])
        else:
            error_message = ('Attributes to calculate alphas ' +
                             'must be of same dimension.')
            raise ValueError(error_message)

        self._alphas = alphas

    @property
    def alphas(self):
        """Compute or return the _alphas attribute."""
        if self._alphas is None:
            self._calculate_alphas()
        return self._alphas

    def constraint_group(self):
        return GenericCHPBlock


class GenericCHPBlock(SimpleBlock):
    r"""
    Block for the relation of the :math:`n` nodes with
    type class:`.GenericCHP`.

    **The following constraints are created:**

    .. _GenericCHP-equations1-10:

    .. math::
        &
        (1)\qquad \dot{H}_F(t) = fuel\ input \\
        &
        (2)\qquad \dot{Q}(t) = heat\ output \\
        &
        (3)\qquad P_{el}(t) = power\ output\\
        &
        (4)\qquad \dot{H}_F(t) = \alpha_0(t) \cdot Y(t) + \alpha_1(t) \cdot
        P_{el,woDH}(t)\\
        &
        (5)\qquad \dot{H}_F(t) = \alpha_0(t) \cdot Y(t) + \alpha_1(t) \cdot
        ( P_{el}(t) + \beta \cdot \dot{Q}(t) )\\
        &
        (6)\qquad \dot{H}_F(t) \leq Y(t) \cdot
        \frac{P_{el, max, woDH}(t)}{\eta_{el,max,woDH}(t)}\\
        &
        (7)\qquad \dot{H}_F(t) \geq Y(t) \cdot
        \frac{P_{el, min, woDH}(t)}{\eta_{el,min,woDH}(t)}\\
        &
        (8)\qquad \dot{H}_{L,FG,max}(t) = \dot{H}_F(t) \cdot
        \dot{H}_{L,FG,sharemax}(t)\\
        &
        (9)\qquad \dot{H}_{L,FG,min}(t) = \dot{H}_F(t) \cdot
        \dot{H}_{L,FG,sharemin}(t)\\
        &
        (10)\qquad P_{el}(t) + \dot{Q}(t) + \dot{H}_{L,FG,max}(t) +
        \dot{Q}_{CW, min}(t) \cdot Y(t) = / \leq \dot{H}_F(t)\\

    where :math:`= / \leq` depends on the CHP being back pressure or not.

    The coefficients :math:`\alpha_0` and :math:`\alpha_1`
    can be determined given the efficiencies maximal/minimal load:

    .. math::
        &
        \eta_{el,max,woDH}(t) = \frac{P_{el,max,woDH}(t)}{\alpha_0(t)
        \cdot Y(t) + \alpha_1(t) \cdot P_{el,max,woDH}(t)}\\
        &
        \eta_{el,min,woDH}(t) = \frac{P_{el,min,woDH}(t)}{\alpha_0(t)
        \cdot Y(t) + \alpha_1(t) \cdot P_{el,min,woDH}(t)}\\


    **For the attribute** :math:`\dot{H}_{L,FG,min}` **being not None**,
    e.g. for a motoric CHP, **the following is created:**

        **Constraint:**

    .. _GenericCHP-equations11:

    .. math::
        &
        (11)\qquad P_{el}(t) + \dot{Q}(t) + \dot{H}_{L,FG,min}(t) +
        \dot{Q}_{CW, min}(t) \cdot Y(t) \geq \dot{H}_F(t)\\[10pt]

    The symbols used are defined as follows (with Variables (V) and Parameters (P)):

    =============================== =============================== ==== =======================
    math. symbol                    attribute                       type explanation
    =============================== =============================== ==== =======================
    :math:`\dot{H}_{F}`             :py:obj:`H_F[n,t]`              V    input of enthalpy
                                                                         through fuel input
    :math:`P_{el}`                  :py:obj:`P[n,t]`                V    provided
                                                                         electric power
    :math:`P_{el,woDH}`             :py:obj:`P_woDH[n,t]`           V    electric power without
                                                                         district heating
    :math:`P_{el,min,woDH}`         :py:obj:`P_min_woDH[n,t]`       P    min. electric power
                                                                         without district heating
    :math:`P_{el,max,woDH}`         :py:obj:`P_max_woDH[n,t]`       P    max. electric power
                                                                         without district heating
    :math:`\dot{Q}`                 :py:obj:`Q[n,t]`                V    provided heat

    :math:`\dot{Q}_{CW, min}`       :py:obj:`Q_CW_min[n,t]`         P    minimal therm. condenser
                                                                         load to cooling water
    :math:`\dot{H}_{L,FG,min}`      :py:obj:`H_L_FG_min[n,t]`       V    flue gas enthalpy loss
                                                                         at min heat extraction
    :math:`\dot{H}_{L,FG,max}`      :py:obj:`H_L_FG_max[n,t]`       V    flue gas enthalpy loss
                                                                         at max heat extraction
    :math:`\dot{H}_{L,FG,sharemin}` :py:obj:`H_L_FG_share_min[n,t]` P    share of flue gas loss
                                                                         at min heat extraction
    :math:`\dot{H}_{L,FG,sharemax}` :py:obj:`H_L_FG_share_max[n,t]` P    share of flue gas loss
                                                                         at max heat extraction
    :math:`Y`                       :py:obj:`Y[n,t]`                V    status variable
                                                                         on/off
    :math:`\alpha_0`                :py:obj:`n.alphas[0][n,t]`      P    coefficient
                                                                         describing efficiency
    :math:`\alpha_1`                :py:obj:`n.alphas[1][n,t]`      P    coefficient
                                                                         describing efficiency
    :math:`\beta`                   :py:obj:`Beta[n,t]`             P    power loss index

    :math:`\eta_{el,min,woDH}`      :py:obj:`Eta_el_min_woDH[n,t]`  P    el. eff. at min. fuel
                                                                         flow w/o distr. heating
    :math:`\eta_{el,max,woDH}`      :py:obj:`Eta_el_max_woDH[n,t]`  P    el. eff. at max. fuel
                                                                         flow w/o distr. heating
    =============================== =============================== ==== =======================

    """
    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """
        Create constraints for GenericCHPBlock.

        Parameters
        ----------
        group : list
            List containing `GenericCHP` objects.
            e.g. groups=[ghcp1, gchp2,..]
        """
        m = self.parent_block()

        if group is None:
            return None

        self.GENERICCHPS = Set(initialize=[n for n in group])

        # variables
        self.H_F = Var(self.GENERICCHPS, m.TIMESTEPS, within=NonNegativeReals)
        self.H_L_FG_max = Var(self.GENERICCHPS, m.TIMESTEPS,
                              within=NonNegativeReals)
        self.H_L_FG_min = Var(self.GENERICCHPS, m.TIMESTEPS,
                              within=NonNegativeReals)
        self.P_woDH = Var(self.GENERICCHPS, m.TIMESTEPS,
                          within=NonNegativeReals)
        self.P = Var(self.GENERICCHPS, m.TIMESTEPS, within=NonNegativeReals)
        self.Q = Var(self.GENERICCHPS, m.TIMESTEPS, within=NonNegativeReals)
        self.Y = Var(self.GENERICCHPS, m.TIMESTEPS, within=Binary)

        # constraint rules
        def _H_flow_rule(block, n, t):
            """Link fuel consumption to component inflow."""
            expr = 0
            expr += self.H_F[n, t]
            expr += - m.flow[list(n.fuel_input.keys())[0], n, t]
            return expr == 0
        self.H_flow = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                 rule=_H_flow_rule)

        def _Q_flow_rule(block, n, t):
            """Link heat flow to component outflow."""
            expr = 0
            expr += self.Q[n, t]
            expr += - m.flow[n, list(n.heat_output.keys())[0], t]
            return expr == 0
        self.Q_flow = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                 rule=_Q_flow_rule)

        def _P_flow_rule(block, n, t):
            """Link power flow to component outflow."""
            expr = 0
            expr += self.P[n, t]
            expr += - m.flow[n, list(n.electrical_output.keys())[0], t]
            return expr == 0
        self.P_flow = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                 rule=_P_flow_rule)

        def _H_F_1_rule(block, n, t):
            """Set P_woDH depending on H_F."""
            expr = 0
            expr += - self.H_F[n, t]
            expr += n.alphas[0][t] * self.Y[n, t]
            expr += n.alphas[1][t] * self.P_woDH[n, t]
            return expr == 0
        self.H_F_1 = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                rule=_H_F_1_rule)

        def _H_F_2_rule(block, n, t):
            """Determine relation between H_F, P and Q."""
            expr = 0
            expr += - self.H_F[n, t]
            expr += n.alphas[0][t] * self.Y[n, t]
            expr += n.alphas[1][t] * (self.P[n, t] + n.Beta[t] * self.Q[n, t])
            return expr == 0
        self.H_F_2 = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                rule=_H_F_2_rule)

        def _H_F_3_rule(block, n, t):
            """Set upper value of operating range via H_F."""
            expr = 0
            expr += self.H_F[n, t]
            expr += - self.Y[n, t] * \
                (list(n.electrical_output.values())[0].P_max_woDH[t] /
                 list(n.electrical_output.values())[0].Eta_el_max_woDH[t])
            return expr <= 0
        self.H_F_3 = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                rule=_H_F_3_rule)

        def _H_F_4_rule(block, n, t):
            """Set lower value of operating range via H_F."""
            expr = 0
            expr += self.H_F[n, t]
            expr += - self.Y[n, t] * \
                (list(n.electrical_output.values())[0].P_min_woDH[t] /
                 list(n.electrical_output.values())[0].Eta_el_min_woDH[t])
            return expr >= 0
        self.H_F_4 = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                rule=_H_F_4_rule)

        def _H_L_FG_max_rule(block, n, t):
            """Set max. flue gas loss as share fuel flow share."""
            expr = 0
            expr += - self.H_L_FG_max[n, t]
            expr += self.H_F[n, t] * \
                list(n.fuel_input.values())[0].H_L_FG_share_max[t]
            return expr == 0
        self.H_L_FG_max_def = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                         rule=_H_L_FG_max_rule)

        def _Q_max_res_rule(block, n, t):
            """Set maximum Q depending on fuel and electrical flow."""
            expr = 0
            expr += self.P[n, t] + self.Q[n, t] + self.H_L_FG_max[n, t]
            expr += list(n.heat_output.values())[0].Q_CW_min[t] * self.Y[n, t]
            expr += - self.H_F[n, t]
            # back-pressure characteristics or one-segment model
            if n.back_pressure is True:
                return expr == 0
            else:
                return expr <= 0
        self.Q_max_res = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                    rule=_Q_max_res_rule)

        def _H_L_FG_min_rule(block, n, t):
            """Set min. flue gas loss as fuel flow share."""
            # minimum flue gas losses e.g. for motoric CHPs
            if getattr(list(n.fuel_input.values())[0],
                       'H_L_FG_share_min', None):
                expr = 0
                expr += - self.H_L_FG_min[n, t]
                expr += self.H_F[n, t] * \
                    list(n.fuel_input.values())[0].H_L_FG_share_min[t]
                return expr == 0
            else:
                return Constraint.Skip
        self.H_L_FG_min_def = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                         rule=_H_L_FG_min_rule)

        def _Q_min_res_rule(block, n, t):
            """Set minimum Q depending on fuel and eletrical flow."""
            # minimum restriction for heat flows e.g. for motoric CHPs
            if getattr(list(n.fuel_input.values())[0],
                       'H_L_FG_share_min', None):
                expr = 0
                expr += self.P[n, t] + self.Q[n, t] + self.H_L_FG_min[n, t]
                expr += list(n.heat_output.values())[0].Q_CW_min[t] \
                    * self.Y[n, t]
                expr += - self.H_F[n, t]
                return expr >= 0
            else:
                return Constraint.Skip
        self.Q_min_res = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                    rule=_Q_min_res_rule)

        def _Y_status_rule(block, n, t):
            if hasattr(m.NonConvexFlow, 'status'):
                return self.Y[n, t] ==\
                    m.NonConvexFlow.status[n, list(n.outputs.keys())[0], t]
            elif hasattr(m.RollingHorizonFlow, 'status'):
                return self.Y[n, t] ==\
                    m.RollingHorizonFlow.status[
                            n, list(n.outputs.keys())[0], t]
            else:
                return Constraint.Skip
        self.Y_status = Constraint(self.GENERICCHPS, m.TIMESTEPS,
                                   rule=_Y_status_rule)

    def _objective_expression(self):
        r"""Objective expression for generic CHPs with no investment.

        Note: This adds nothing as variable costs are already
        added in the Block :class:`Flow`.
        """
        if not hasattr(self, 'GENERICCHPS'):
            return 0

        return 0


class GenericPowerPlant(network.Transformer):
    r"""
    Component `GenericCHP` to model combined heat and power plants.

    Can be used to model (combined cycle) extraction or back-pressure turbines
    and used a mixed-integer linear formulation. Thus, it induces more
    computational effort than the `ExtractionTurbineCHP` for the
    benefit of higher accuracy.

    The full set of equations is described in:
    Mollenhauer, E., Christidis, A. & Tsatsaronis, G.
    Evaluation of an energy- and exergy-based generic modeling
    approach of combined heat and power plants
    Int J Energy Environ Eng (2016) 7: 167.
    https://doi.org/10.1007/s40095-016-0204-6

    For a general understanding of (MI)LP CHP representation, see:
    Fabricio I. Salgado, P.
    Short - Term Operation Planning on Cogeneration Systems: A Survey
    Electric Power Systems Research (2007)
    Electric Power Systems Research
    Volume 78, Issue 5, May 2008, Pages 835-848
    https://doi.org/10.1016/j.epsr.2007.06.001

    Note
    ----
    An adaption for the flow parameter `H_L_FG_share_max` has been made to
    set the flue gas losses at maximum heat extraction `H_L_FG_max` as share of
    the fuel flow `H_F` e.g. for combined cycle extraction turbines.
    The flow parameter `H_L_FG_share_min` can be used to set the flue gas
    losses at minimum heat extraction `H_L_FG_min` as share of
    the fuel flow `H_F` e.g. for motoric CHPs.
    The boolean component parameter `back_pressure` can be set to model
    back-pressure characteristics.

    Also have a look at the examples on how to use it.

    Parameters
    ----------
    fuel_input : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the fuel input.
    electrical_output : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the electrical output. Related parameters like `P_max` are
        passed as attributes of the `oemof.Flow` object.
    heat_output : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the heat output. Related parameters like `Q_CW_min` are passed as
        attributes of the `oemof.Flow` object.
    Beta : list of numerical values
        Beta values in same dimension as all other parameters (length of
        optimization period).
    back_pressure : boolean
        Flag to use back-pressure characteristics. Set to `True` and
        `Q_CW_min` to zero for back-pressure turbines. See paper above for more
        information.

    Note
    ----
    The following sets, variables, constraints and objective parts are created
     * :py:class:`~oemof.solph.components.GenericCHPBlock`

    Examples
    --------
    >>> from oemof import solph
    >>> bel = solph.Bus(label='electricityBus')
    >>> bth = solph.Bus(label='heatBus')
    >>> bgas = solph.Bus(label='commodityBus')
    >>> ccet = solph.components.GenericCHP(
    ...    label='combined_cycle_extraction_turbine',
    ...    fuel_input={bgas: solph.Flow(
    ...        H_L_FG_share_max=[0.183])},
    ...    electrical_output={bel: solph.Flow(
    ...        P_max=[155.946],
    ...        P_min=[68.787],
    ...        Eta_el_max=[0.525],
    ...        Eta_el_min=[0.444])},
    ...    heat_output={bth: solph.Flow(
    ...        Q_CW_min=[10.552])},
    ...    Beta=[0.122], back_pressure=False)
    >>> type(ccet)
    <class 'oemof.solph.components.GenericCHP'>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fuel_input = kwargs.get('fuel_input')
        self.electrical_output = kwargs.get('electrical_output')
        self._alphas = None

        # map specific flows to standard API
        fuel_bus = list(self.fuel_input.keys())[0]
        fuel_flow = list(self.fuel_input.values())[0]
        fuel_bus.outputs.update({self: fuel_flow})

        self.outputs.update(kwargs.get('electrical_output'))

    def _calculate_alphas(self):
        """
        Calculate alpha coefficients.

        A system of linear equations is created from passed capacities and
        efficiencies and solved to calculate both coefficients.
        """
        alphas = [[], []]

        eb = list(self.electrical_output.keys())[0]

        attrs = [self.electrical_output[eb].P_min,
                 self.electrical_output[eb].Eta_el_min,
                 self.electrical_output[eb].P_max,
                 self.electrical_output[eb].Eta_el_max]

        length = [len(a) for a in attrs if not isinstance(a, (int, float))]
        max_length = max(length)

        if all(len(a) == max_length for a in attrs):
            if max_length == 0:
                max_length += 1  # increment dimension for scalars from 0 to 1
            for i in range(0, max_length):
                A = np.array([[1, self.electrical_output[eb].P_min[i]],
                              [1, self.electrical_output[eb].P_max[i]]])
                b = np.array([self.electrical_output[eb].P_min[i] /
                              self.electrical_output[eb].Eta_el_min[i],
                              self.electrical_output[eb].P_max[i] /
                              self.electrical_output[eb].Eta_el_max[i]])
                x = np.linalg.solve(A, b)
                alphas[0].append(x[0])
                alphas[1].append(x[1])
        else:
            error_message = ('Attributes to calculate alphas ' +
                             'must be of same dimension.')
            raise ValueError(error_message)

        self._alphas = alphas

    @property
    def alphas(self):
        """Compute or return the _alphas attribute."""
        if self._alphas is None:
            self._calculate_alphas()
        return self._alphas

    def constraint_group(self):
        return GenericPowerPlantBlock


class GenericPowerPlantBlock(SimpleBlock):
    r"""
    Block for the relation of the :math:`n` nodes with
    type class:`.GenericCHP`.

    **The following constraints are created:**

    .. _GenericCHP-equations1-10:

    .. math::
        &
        (1)\qquad \dot{H}_F(t) = fuel\ input \\
        &
        (2)\qquad \dot{Q}(t) = heat\ output \\
        &
        (3)\qquad P_{el}(t) = power\ output\\
        &
        (4)\qquad \dot{H}_F(t) = \alpha_0(t) \cdot Y(t) + \alpha_1(t) \cdot
        P_{el}(t)\\
        &
        (5)\qquad \dot{H}_F(t) = \alpha_0(t) \cdot Y(t) + \alpha_1(t) \cdot
        ( P_{el}(t) + \beta \cdot \dot{Q}(t) )\\
        &
        (6)\qquad \dot{H}_F(t) \leq Y(t) \cdot
        \frac{P_{el, max, woDH}(t)}{\eta_{el,max}(t)}\\
        &
        (7)\qquad \dot{H}_F(t) \geq Y(t) \cdot
        \frac{P_{el, min, woDH}(t)}{\eta_{el,min}(t)}\\
        &
        (8)\qquad \dot{H}_{L,FG,max}(t) = \dot{H}_F(t) \cdot
        \dot{H}_{L,FG,sharemax}(t)\\
        &
        (9)\qquad \dot{H}_{L,FG,min}(t) = \dot{H}_F(t) \cdot
        \dot{H}_{L,FG,sharemin}(t)\\
        &
        (10)\qquad P_{el}(t) + \dot{Q}(t) + \dot{H}_{L,FG,max}(t) +
        \dot{Q}_{CW, min}(t) \cdot Y(t) = / \leq \dot{H}_F(t)\\

    where :math:`= / \leq` depends on the CHP being back pressure or not.

    The coefficients :math:`\alpha_0` and :math:`\alpha_1`
    can be determined given the efficiencies maximal/minimal load:

    .. math::
        &
        \eta_{el,max}(t) = \frac{P_{el,max}(t)}{\alpha_0(t)
        \cdot Y(t) + \alpha_1(t) \cdot P_{el,max}(t)}\\
        &
        \eta_{el,min}(t) = \frac{P_{el,min}(t)}{\alpha_0(t)
        \cdot Y(t) + \alpha_1(t) \cdot P_{el,min}(t)}\\


    **For the attribute** :math:`\dot{H}_{L,FG,min}` **being not None**,
    e.g. for a motoric CHP, **the following is created:**

        **Constraint:**

    .. _GenericCHP-equations11:

    .. math::
        &
        (11)\qquad P_{el}(t) + \dot{Q}(t) + \dot{H}_{L,FG,min}(t) +
        \dot{Q}_{CW, min}(t) \cdot Y(t) \geq \dot{H}_F(t)\\[10pt]

    The symbols used are defined as follows (with Variables (V) and Parameters (P)):

    =============================== =============================== ==== =======================
    math. symbol                    attribute                       type explanation
    =============================== =============================== ==== =======================
    :math:`\dot{H}_{F}`             :py:obj:`H_F[n,t]`              V    input of enthalpy
                                                                         through fuel input
    :math:`P_{el}`                  :py:obj:`P[n,t]`                V    provided
                                                                         electric power
    :math:`Y`                       :py:obj:`Y[n,t]`                V    status variable
                                                                         on/off
    :math:`\alpha_0`                :py:obj:`n.alphas[0][n,t]`      P    coefficient
                                                                         describing efficiency
    :math:`\alpha_1`                :py:obj:`n.alphas[1][n,t]`      P    coefficient
                                                                         describing efficiency

    :math:`\eta_{el,min}`      :py:obj:`Eta_el_min[n,t]`  P    el. eff. at min. fuel
                                                                         flow
    :math:`\eta_{el,max}`      :py:obj:`Eta_el_max[n,t]`  P    el. eff. at max. fuel
                                                                         flow
    =============================== =============================== ==== =======================

    """
    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """
        Create constraints for GenericPlantBlock.

        Parameters
        ----------
        group : list
            List containing `GenericPlant` objects.
            e.g. groups=[ghcp1, gchp2,..]
        """
        m = self.parent_block()

        if group is None:
            return None

        self.GENERICPLANT = Set(initialize=[n for n in group])

        # variables
        self.H_F = Var(self.GENERICPLANT, m.TIMESTEPS, within=NonNegativeReals)
        self.P = Var(self.GENERICPLANT, m.TIMESTEPS, within=NonNegativeReals)
        self.Y = Var(self.GENERICPLANT, m.TIMESTEPS, within=Binary)

        # constraint rules
        def _H_flow_rule(block, n, t):
            """Link fuel consumption to component inflow."""
            expr = 0
            expr += self.H_F[n, t]
            expr += - m.flow[list(n.fuel_input.keys())[0], n, t]
            return expr == 0
        self.H_flow = Constraint(self.GENERICPLANT, m.TIMESTEPS,
                                 rule=_H_flow_rule)

        def _P_flow_rule(block, n, t):
            """Link power flow to component outflow."""
            expr = 0
            expr += self.P[n, t]
            expr += - m.flow[n, list(n.electrical_output.keys())[0], t]
            return expr == 0
        self.P_flow = Constraint(self.GENERICPLANT, m.TIMESTEPS,
                                 rule=_P_flow_rule)

        def _H_F_1_rule(block, n, t):
            """Determine relation between H_F, and P."""
            expr = 0
            expr += - self.H_F[n, t]
            expr += n.alphas[0][t] * self.Y[n, t]
            expr += n.alphas[1][t] * (self.P[n, t])
            return expr == 0
        self.H_F_1 = Constraint(self.GENERICPLANT, m.TIMESTEPS,
                                rule=_H_F_1_rule)

        def _H_F_2_rule(block, n, t):
            """Set upper value of operating range via H_F."""
            expr = 0
            expr += self.H_F[n, t]
            expr += - self.Y[n, t] * \
                (list(n.electrical_output.values())[0].P_max[t] /
                 list(n.electrical_output.values())[0].Eta_el_max[t])
            return expr <= 0
        self.H_F_2 = Constraint(self.GENERICPLANT, m.TIMESTEPS,
                                rule=_H_F_2_rule)

        def _H_F_3_rule(block, n, t):
            """Set lower value of operating range via H_F."""
            expr = 0
            expr += self.H_F[n, t]
            expr += - self.Y[n, t] * \
                (list(n.electrical_output.values())[0].P_min[t] /
                 list(n.electrical_output.values())[0].Eta_el_min[t])
            return expr >= 0
        self.H_F_3 = Constraint(self.GENERICPLANT, m.TIMESTEPS,
                                rule=_H_F_3_rule)

        def _Y_status_rule(block, n, t):
            if hasattr(m.NonConvexFlow, 'status'):
                return self.Y[n, t] ==\
                    m.NonConvexFlow.status[n, list(n.outputs.keys())[0], t]
            elif hasattr(m.RollingHorizonFlow, 'status'):
                return self.Y[n, t] ==\
                    m.RollingHorizonFlow.status[
                            n, list(n.outputs.keys())[0], t]
            else:
                return Constraint.Skip
        self.Y_status = Constraint(self.GENERICPLANT, m.TIMESTEPS,
                                   rule=_Y_status_rule)

    def _objective_expression(self):
        r"""Objective expression for generic Plant with no investment.

        Note: This adds nothing as variable costs are already
        added in the Block :class:`Flow`.
        """
        if not hasattr(self, 'GENERICPLANT'):
            return 0

        return 0


class ElectricalBus(Bus):
    r"""A electrical bus object. Every node has to be connected to Bus. This
    Bus is used in combination with ElectricalLine objects for linear optimal
    power flow (lopf) calculations.

    Parameters
    ----------
    slack: boolean
        If True Bus is slack bus for network
    v_max: numeric
        Maximum value of voltage angle at electrical bus
    v_min: numeric
        Mininum value of voltag angle at electrical bus

    Note: This component is experimental. Use it with care.

    Notes
    -----
    The following sets, variables, constraints and objective parts are created
     * :py:class:`~oemof.solph.blocks.Bus`
    The objects are also used inside:
     * :py:class:`~oemof.solph.custom.ElectricalLine`

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slack = kwargs.get('slack', False)
        self.v_max = kwargs.get('v_max', 1000)
        self.v_min = kwargs.get('v_min', -1000)


class ElectricalLine(Flow):
    r"""An ElectricalLine to be used in linear optimal power flow calculations.
    based on angle formulation. Check out the Notes below before using this
    component!

    Parameters
    ----------
    reactance : float or array of floats
        Reactance of the line to be modelled

    Note: This component is experimental. Use it with care.

    Notes
    -----
    * To use this object the connected buses need to be of the type
      :py:class:`~oemof.solph.custom.ElectricalBus`.
    * It does not work together with flows that have set the attr.`nonconvex`,
      i.e. unit commitment constraints are not possible
    * Input and output of this component are set equal, therefore just use
      either only the input or the output to parameterize.
    * Default attribute `min` of in/outflows is overwritten by -1 if not set
      differently by the user

    The following sets, variables, constraints and objective parts are created
     * :py:class:`~oemof.solph.custom.ElectricalLineBlock`

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reactance = sequence(kwargs.get('reactance', 0.00001))

        # set input / output flow values to -1 by default if not set by user
        if self.nonconvex is not None:
            raise ValueError(
                ("Attribute `nonconvex` must be None for " +
                 "component `ElectricalLine` from {} to {}!").format(
                    self.input, self.output))
        if self.min is None:
            self.min = -1
        # to be used in grouping for all bidi flows
        self.bidirectional = True

    def constraint_group(self):
        return ElectricalLineBlock


class ElectricalLineBlock(SimpleBlock):
    r"""Block for the linear relation of nodes with type
    class:`.ElectricalLine`

    Note: This component is experimental. Use it with care.

    **The following constraints are created:**

    Linear relation :attr:`om.ElectricalLine.electrical_flow[n,t]`
        .. math::
            flow(n, o, t) =  1 / reactance(n, t) \\cdot ()
            voltage_angle(i(n), t) - volatage_angle(o(n), t), \\
            \forall t \\in \\textrm{TIMESTEPS}, \\
            \forall n \\in \\textrm{ELECTRICAL\_LINES}.

    TODO: Add equate constraint of flows

    **The following variable are created:**

    TODO: Add voltage angle variable

    TODO: Add fix slack bus voltage angle to zero constraint / bound

    TODO: Add tests
    """

    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """ Creates the linear constraint for the class:`ElectricalLine`
        block.

        Parameters
        ----------
        group : list
            List of oemof.solph.ElectricalLine (eline) objects for which
            the linear relation of inputs and outputs is created
            e.g. group = [eline1, eline2, ...]. The components inside the
            list need to hold a attribute `reactance` of type Sequence
            containing the reactance of the line.
        """
        if group is None:
            return None

        m = self.parent_block()

        # create voltage angle variables
        self.ELECTRICAL_BUSES = Set(initialize=[n for n in m.es.nodes
                                    if isinstance(n, ElectricalBus)])

        def _voltage_angle_bounds(block, b, t):
            return b.v_min, b.v_max
        self.voltage_angle = Var(self.ELECTRICAL_BUSES, m.TIMESTEPS,
                                    bounds=_voltage_angle_bounds)

        if True not in [b.slack for b in self.ELECTRICAL_BUSES]:
            # TODO: Make this robust to select the same slack bus for
            # the same problems
            bus = [b for b in self.ELECTRICAL_BUSES][0]
            logging.info(
                "No slack bus set,setting bus {0} as slack bus".format(
                    bus.label))
            bus.slack = True

        def _voltage_angle_relation(block):
            for t in m.TIMESTEPS:
                for n in group:
                    if n.input.slack is True:
                        self.voltage_angle[n.output, t].value = 0
                        self.voltage_angle[n.output, t].fix()
                    try:
                        lhs = m.flow[n.input, n.output, t]
                        rhs = 1 / n.reactance[t] * (
                            self.voltage_angle[n.input, t] -
                            self.voltage_angle[n.output, t])
                    except:
                        raise ValueError("Error in constraint creation",
                                         "of node {}".format(n.label))
                    block.electrical_flow.add((n, t), (lhs == rhs))

        self.electrical_flow = Constraint(group, m.TIMESTEPS, noruleinit=True)

        self.electrical_flow_build = BuildAction(
                                         rule=_voltage_angle_relation)


class Link(Transformer):
    """A Link object with 1...2 inputs and 1...2 outputs.

    Parameters
    ----------
    conversion_factors : dict
        Dictionary containing conversion factors for conversion of each flow.
        Keys are the connected tuples (input, output) bus objects.
        The dictionary values can either be a scalar or a sequence with length
        of time horizon for simulation.

    Note: This component is experimental. Use it with care.

    Notes
    -----
    The sets, variables, constraints and objective parts are created
     * :py:class:`~oemof.solph.custom.LinkBlock`

    Examples
    --------

    >>> from oemof import solph
    >>> bel0 = solph.Bus(label="el0")
    >>> bel1 = solph.Bus(label="el1")

    >>> link = solph.custom.Link(
    ...    label="transshipment_link",
    ...    inputs={bel0: solph.Flow(), bel1: solph.Flow()},
    ...    outputs={bel0: solph.Flow(), bel1: solph.Flow()},
    ...    conversion_factors={(bel0, bel1): 0.92, (bel1, bel0): 0.99})
    >>> print(sorted([x[1][5] for x in link.conversion_factors.items()]))
    [0.92, 0.99]

    >>> type(link)
    <class 'oemof.solph.custom.Link'>

    >>> sorted([str(i) for i in link.inputs])
    ['el0', 'el1']

    >>> link.conversion_factors[(bel0, bel1)][3]
    0.92
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.inputs) > 2 or len(self.outputs) > 2:
            raise ValueError("Component `Link` must not have more than \
                             2 inputs and 2 outputs!")

        self.conversion_factors = {
            k: sequence(v)
            for k, v in kwargs.get('conversion_factors', {}).items()}

    def constraint_group(self):
        return LinkBlock


class LinkBlock(SimpleBlock):
    r"""Block for the relation of nodes with type
    :class:`~oemof.solph.custom.Link`

    Note: This component is experimental. Use it with care.

    **The following constraints are created:**

    TODO: Add description for constraints
    TODO: Add tests

    """
    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """ Creates the relation for the class:`Link`.

        Parameters
        ----------
        group : list
            List of oemof.solph.custom.Link objects for which
            the relation of inputs and outputs is createdBuildAction
            e.g. group = [link1, link2, link3, ...]. The components inside
            the list need to hold an attribute `conversion_factors` of type
            dict containing the conversion factors for all inputs to outputs.
        """
        if group is None:
            return None

        m = self.parent_block()

        all_conversions = {}
        for n in group:
            all_conversions[n] = {
                            k: v for k, v in n.conversion_factors.items()}

        def _input_output_relation(block):
            for t in m.TIMESTEPS:
                for n, conversion in all_conversions.items():
                    for cidx, c in conversion.items():
                        try:
                            expr = (m.flow[n, cidx[1], t] ==
                                    c[t] * m.flow[cidx[0], n, t])
                        except ValueError:
                            raise ValueError(
                                "Error in constraint creation",
                                "from: {0}, to: {1}, via: {3}".format(
                                    cidx[0], cidx[1], n))
                        block.relation.add((n, cidx[0], cidx[1], t), (expr))

        self.relation = Constraint(
            [(n, cidx[0], cidx[1], t)
             for t in m.TIMESTEPS
             for n, conversion in all_conversions.items()
             for cidx, c in conversion.items()], noruleinit=True)
        self.relation_build = BuildAction(rule=_input_output_relation)


class GenericCAES(Transformer):
    """
    Component `GenericCAES` to model arbitrary compressed air energy storages.

    The full set of equations is described in:
    Kaldemeyer, C.; Boysen, C.; Tuschy, I.
    A Generic Formulation of Compressed Air Energy Storage as
    Mixed Integer Linear Program – Unit Commitment of Specific
    Technical Concepts in Arbitrary Market Environments
    Materials Today: Proceedings 00 (2018) 0000–0000
    [currently in review]

    Parameters
    ----------
    electrical_input : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the electrical input.
    fuel_input : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the fuel input.
    electrical_output : dict
        Dictionary with key-value-pair of `oemof.Bus` and `oemof.Flow` object
        for the electrical output.

    Note: This component is experimental. Use it with care.

    Notes
    -----
    The following sets, variables, constraints and objective parts are created
     * :py:class:`~oemof.solph.blocks.GenericCAES`

    TODO: Add description for constraints. See referenced paper until then!

    Examples
    --------

    >>> from oemof import solph
    >>> bel = solph.Bus(label='bel')
    >>> bth = solph.Bus(label='bth')
    >>> bgas = solph.Bus(label='bgas')
    >>> # dictionary with parameters for a specific CAES plant
    >>> concept = {
    ...    'cav_e_in_b': 0,
    ...    'cav_e_in_m': 0.6457267578,
    ...    'cav_e_out_b': 0,
    ...    'cav_e_out_m': 0.3739636077,
    ...    'cav_eta_temp': 1.0,
    ...    'cav_level_max': 211.11,
    ...    'cmp_p_max_b': 86.0918959849,
    ...    'cmp_p_max_m': 0.0679999932,
    ...    'cmp_p_min': 1,
    ...    'cmp_q_out_b': -19.3996965679,
    ...    'cmp_q_out_m': 1.1066036114,
    ...    'cmp_q_tes_share': 0,
    ...    'exp_p_max_b': 46.1294016678,
    ...    'exp_p_max_m': 0.2528340303,
    ...    'exp_p_min': 1,
    ...    'exp_q_in_b': -2.2073411014,
    ...    'exp_q_in_m': 1.129249765,
    ...    'exp_q_tes_share': 0,
    ...    'tes_eta_temp': 1.0,
    ...    'tes_level_max': 0.0}
    >>> # generic compressed air energy storage (caes) plant
    >>> caes = solph.custom.GenericCAES(
    ...    label='caes',
    ...    electrical_input={bel: solph.Flow()},
    ...    fuel_input={bgas: solph.Flow()},
    ...    electrical_output={bel: solph.Flow()},
    ...    params=concept, fixed_costs=0)
    >>> type(caes)
    <class 'oemof.solph.custom.GenericCAES'>
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.electrical_input = kwargs.get('electrical_input')
        self.fuel_input = kwargs.get('fuel_input')
        self.electrical_output = kwargs.get('electrical_output')
        self.params = kwargs.get('params')

        # map specific flows to standard API
        self.inputs.update(kwargs.get('electrical_input'))
        self.inputs.update(kwargs.get('fuel_input'))
        self.outputs.update(kwargs.get('electrical_output'))

    def constraint_group(self):
        return GenericCAESBlock


class GenericCAESBlock(SimpleBlock):
    r"""Block for nodes of class:`.GenericCAES`.

    Note: This component is experimental. Use it with care.

    **The following constraints are created:**

    .. _GenericCAES-equations:

    .. math::
        &
        (1) \qquad P_{cmp}(t) = electrical\_input (t)
            \quad \forall t \in T \\
        &
        (2) \qquad P_{cmp\_max}(t) = m_{cmp\_max} \cdot CAS_{fil}(t-1)
            + b_{cmp\_max}
            \quad \forall t \in\left[1, t_{max}\right] \\
        &
        (3) \qquad P_{cmp\_max}(t) = b_{cmp\_max}
            \quad \forall t \notin\left[1, t_{max}\right] \\
        &
        (4) \qquad P_{cmp}(t) \leq P_{cmp\_max}(t)
            \quad \forall t \in T  \\
        &
        (5) \qquad P_{cmp}(t) \geq P_{cmp\_min} \cdot ST_{cmp}(t)
            \quad \forall t \in T  \\
        &
        (6) \qquad P_{cmp}(t) = m_{cmp\_max} \cdot CAS_{fil\_max}
            + b_{cmp\_max} \cdot ST_{cmp}(t)
            \quad \forall t \in T \\
        &
        (7) \qquad \dot{Q}_{cmp}(t) =
            m_{cmp\_q} \cdot P_{cmp}(t) + b_{cmp\_q} \cdot ST_{cmp}(t)
            \quad \forall t \in T  \\
        &
        (8) \qquad \dot{Q}_{cmp}(t) = \dot{Q}_{cmp_out}(t)
            + \dot{Q}_{tes\_in}(t)
            \quad \forall t \in T \\
        &
        (9) \qquad r_{cmp\_tes} \cdot\dot{Q}_{cmp\_out}(t) =
            \left(1-r_{cmp\_tes}\right) \dot{Q}_{tes\_in}(t)
            \quad \forall t \in T \\
        &
        (10) \quad\; P_{exp}(t) = electrical\_output (t)
             \quad \forall t \in T \\
        &
        (11) \quad\; P_{exp\_max}(t) = m_{exp\_max} CAS_{fil}(t-1)
             + b_{exp\_max}
             \quad \forall t \in\left[1, t_{\max }\right] \\
        &
        (12) \quad\; P_{exp\_max}(t) = b_{exp\_max}
             \quad \forall t \notin\left[1, t_{\max }\right] \\
        &
        (13) \quad\; P_{exp}(t) \leq P_{exp\_max}(t)
             \quad \forall t \in T \\
        &
        (14) \quad\; P_{exp}(t) \geq P_{exp\_min}(t) \cdot ST_{exp}(t)
             \quad \forall t \in T \\
        &
        (15) \quad\; P_{exp}(t) \leq m_{exp\_max} \cdot CAS_{fil\_max}
             + b_{exp\_max} \cdot ST_{exp}(t)
             \quad \forall t \in T \\
        &
        (16) \quad\; \dot{Q}_{exp}(t) = m_{exp\_q} \cdot P_{exp}(t)
             + b_{cxp\_q} \cdot ST_{cxp}(t)
             \quad \forall t \in T \\
        &
        (17) \quad\; \dot{Q}_{exp\_in}(t) = fuel\_input(t)
             \quad \forall t \in T \\
        &
        (18) \quad\; \dot{Q}_{exp}(t) = \dot{Q}_{exp\_in}(t)
             + \dot{Q}_{tes\_out}(t)+\dot{Q}_{cxp\_add}(t)
             \quad \forall t \in T \\
        &
        (19) \quad\; r_{exp\_tes} \cdot \dot{Q}_{exp\_in}(t) =
             (1 - r_{exp\_tes})(\dot{Q}_{tes\_out}(t) + \dot{Q}_{exp\_add}(t))
             \quad \forall t \in T \\
        &
        (20) \quad\; \dot{E}_{cas\_in}(t) = m_{cas\_in}\cdot P_{cmp}(t)
             + b_{cas\_in}\cdot ST_{cmp}(t)
             \quad \forall t \in T \\
        &
        (21) \quad\; \dot{E}_{cas\_out}(t) = m_{cas\_out}\cdot P_{cmp}(t)
             + b_{cas\_out}\cdot ST_{cmp}(t)
             \quad \forall t \in T \\
        &
        (22) \quad\; \eta_{cas\_tmp} \cdot CAS_{fil}(t) = CAS_{fil}(t-1)
             + \tau\left(\dot{E}_{cas\_in}(t) - \dot{E}_{cas\_out}(t)\right)
             \quad \forall t \in\left[1, t_{max}\right] \\
         &
        (23) \quad\; \eta_{cas\_tmp} \cdot CAS_{fil}(t) =
             \tau\left(\dot{E}_{cas\_in}(t) - \dot{E}_{cas\_out}(t)\right)
             \quad \forall t \notin\left[1, t_{max}\right] \\
        &
        (24) \quad\; CAS_{fil}(t) \leq CAS_{fil\_max}
             \quad \forall t \in T \\
        &
        (25) \quad\; TES_{fil}(t) = TES_{fil}(t-1)
             + \tau\left(\dot{Q}_{tes\_in}(t)
             - \dot{Q}_{tes\_out}(t)\right)
             \quad \forall t \in\left[1, t_{max}\right] \\
         &
        (26) \quad\; TES_{fil}(t) =
             \tau\left(\dot{Q}_{tes\_in}(t)
             - \dot{Q}_{tes\_out}(t)\right)
             \quad \forall t \notin\left[1, t_{max}\right] \\
        &
        (27) \quad\; TES_{fil}(t) \leq TES_{fil\_max}
             \quad \forall t \in T \\
        &


    **Table: Symbols and attribute names of variables and parameters**

    .. csv-table:: Variables (V) and Parameters (P)
        :header: "symbol", "attribute", "type", "explanation"
        :widths: 1, 1, 1, 1

        ":math:`ST_{cmp}` ", ":py:obj:`cmp_st[n,t]` ", "V", "Status of compression"
        ":math:`{P}_{cmp}` ", ":py:obj:`cmp_p[n,t]`", "V", "Compression power"
        ":math:`{P}_{cmp\_max}`", ":py:obj:`cmp_p_max[n,t]`", "V", "Max. compression power"
        ":math:`\dot{Q}_{cmp}` ", ":py:obj:`cmp_q_out_sum[n,t]`", "V", "Summed heat flow in compression"
        ":math:`\dot{Q}_{cmp\_out}` ", ":py:obj:`cmp_q_waste[n,t]`", "V", "Waste heat flow from compression"
        ":math:`ST_{exp}(t)`", ":py:obj:`exp_st[n,t]`", "V", "Status of expansion (binary)"
        ":math:`P_{exp}(t)`", ":py:obj:`exp_p[n,t]`", "V", "Expansion power"
        ":math:`P_{exp\_max}(t)`", ":py:obj:`exp_p_max[n,t]`", "V", "Max. expansion power"
        ":math:`\dot{Q}_{exp}(t)`", ":py:obj:`exp_q_in_sum[n,t]`", "V", "Summed heat flow in expansion"
        ":math:`\dot{Q}_{exp\_in}(t)`", ":py:obj:`exp_q_fuel_in[n,t]`", "V", "Heat (external) flow into expansion"
        ":math:`\dot{Q}_{exp\_add}(t)`", ":py:obj:`exp_q_add_in[n,t]`", "V", "Additional heat flow into expansion"
        ":math:`CAV_{fil}(t)`", ":py:obj:`cav_level[n,t]`", "V", "Filling level if CAE"
        ":math:`\dot{E}_{cas\_in}(t)`", ":py:obj:`cav_e_in[n,t]`", "V", "Exergy flow into CAS"
        ":math:`\dot{E}_{cas\_out}(t)`", ":py:obj:`cav_e_out[n,t]`", "V", "Exergy flow from CAS"
        ":math:`TES_{fil}(t)`", ":py:obj:`tes_level[n,t]`", "V", "Filling level of Thermal Energy Storage (TES)"
        ":math:`\dot{Q}_{tes\_in}(t)`", ":py:obj:`tes_e_in[n,t]`", "V", "Heat flow into TES"
        ":math:`\dot{Q}_{tes\_out}(t)`", ":py:obj:`tes_e_out[n,t]`", "V", "Heat flow from TES"
        ":math:`b_{cmp\_max}`", ":py:obj:`cmp_p_max_b[n,t]`", "P", "Specific y-intersection"
        ":math:`b_{cmp\_q}`", ":py:obj:`cmp_q_out_b[n,t]`", "P", "Specific y-intersection"
        ":math:`b_{exp\_max}`", ":py:obj:`exp_p_max_b[n,t]`", "P", "Specific y-intersection"
        ":math:`b_{exp\_q}`", ":py:obj:`exp_q_in_b[n,t]`", "P", "Specific y-intersection"
        ":math:`b_{cas\_in}`", ":py:obj:`cav_e_in_b[n,t]`", "P", "Specific y-intersection"
        ":math:`b_{cas\_out}`", ":py:obj:`cav_e_out_b[n,t]`", "P", "Specific y-intersection"
        ":math:`m_{cmp\_max}`", ":py:obj:`cmp_p_max_m[n,t]`", "P", "Specific slope"
        ":math:`m_{cmp\_q}`", ":py:obj:`cmp_q_out_m[n,t]`", "P", "Specific slope"
        ":math:`m_{exp\_max}`", ":py:obj:`exp_p_max_m[n,t]`", "P", "Specific slope"
        ":math:`m_{exp\_q}`", ":py:obj:`exp_q_in_m[n,t]`", "P", "Specific slope"
        ":math:`m_{cas\_in}`", ":py:obj:`cav_e_in_m[n,t]`", "P", "Specific slope"
        ":math:`m_{cas\_out}`", ":py:obj:`cav_e_out_m[n,t]`", "P", "Specific slope"
        ":math:`P_{cmp\_min}`", ":py:obj:`cmp_p_min[n,t]`", "P", "Min. compression power"
        ":math:`r_{cmp\_tes}`", ":py:obj:`cmp_q_tes_share[n,t]`", "P", "Ratio between waste heat flow and heat flow into TES"
        ":math:`r_{exp\_tes}`", ":py:obj:`exp_q_tes_share[n,t]`", "P", "Ratio between external heat flow into expansion and heat flows from TES and additional source"
        ":math:`\tau`", ":py:obj:`m.timeincrement[n,t]`", "P", "Time interval length"
        ":math:`TES_{fil\_max}`", ":py:obj:`tes_level_max[n,t]`", "P", "Max. filling level of TES"
        ":math:`CAS_{fil\_max}`", ":py:obj:`cav_level_max[n,t]`", "P", "Max. filling level of TES"
        ":math:`\tau`", ":py:obj:`cav_eta_tmp[n,t]`", "P", "Temporal efficiency (loss factor to take intertemporal losses into account)"
        ":math:`electrical\_input`", ":py:obj:`flow[list(n.electrical_input.keys())[0], n, t]`", "P", "Electr. power input into compression"
        ":math:`electrical\_output`", ":py:obj:`flow[n, list(n.electrical_output.keys())[0], t]`", "P", "Electr. power output of expansion"
        ":math:`fuel\_input`", ":py:obj:`flow[list(n.fuel_input.keys())[0], n, t]`", "P", "Heat input (external) into Expansion"

    """

    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """
        Create constraints for GenericCAESBlock.

        Parameters
        ----------
        group : list
            List containing `.GenericCAES` objects.
            e.g. groups=[gcaes1, gcaes2,..]
        """
        m = self.parent_block()

        if group is None:
            return None

        self.GENERICCAES = Set(initialize=[n for n in group])

        # Compression: Binary variable for operation status
        self.cmp_st = Var(self.GENERICCAES, m.TIMESTEPS, within=Binary)

        # Compression: Realized capacity
        self.cmp_p = Var(self.GENERICCAES, m.TIMESTEPS,
                         within=NonNegativeReals)

        # Compression: Max. Capacity
        self.cmp_p_max = Var(self.GENERICCAES, m.TIMESTEPS,
                             within=NonNegativeReals)

        # Compression: Heat flow
        self.cmp_q_out_sum = Var(self.GENERICCAES, m.TIMESTEPS,
                                 within=NonNegativeReals)

        # Compression: Waste heat
        self.cmp_q_waste = Var(self.GENERICCAES, m.TIMESTEPS,
                               within=NonNegativeReals)

        # Expansion: Binary variable for operation status
        self.exp_st = Var(self.GENERICCAES, m.TIMESTEPS, within=Binary)

        # Expansion: Realized capacity
        self.exp_p = Var(self.GENERICCAES, m.TIMESTEPS,
                         within=NonNegativeReals)

        # Expansion: Max. Capacity
        self.exp_p_max = Var(self.GENERICCAES, m.TIMESTEPS,
                             within=NonNegativeReals)

        # Expansion: Heat flow of natural gas co-firing
        self.exp_q_in_sum = Var(self.GENERICCAES, m.TIMESTEPS,
                                within=NonNegativeReals)

        # Expansion: Heat flow of natural gas co-firing
        self.exp_q_fuel_in = Var(self.GENERICCAES, m.TIMESTEPS,
                                 within=NonNegativeReals)

        # Expansion: Heat flow of additional firing
        self.exp_q_add_in = Var(self.GENERICCAES, m.TIMESTEPS,
                                within=NonNegativeReals)

        # Cavern: Filling levelh
        self.cav_level = Var(self.GENERICCAES, m.TIMESTEPS,
                             within=NonNegativeReals)

        # Cavern: Energy inflow
        self.cav_e_in = Var(self.GENERICCAES, m.TIMESTEPS,
                            within=NonNegativeReals)

        # Cavern: Energy outflow
        self.cav_e_out = Var(self.GENERICCAES, m.TIMESTEPS,
                             within=NonNegativeReals)

        # TES: Filling levelh
        self.tes_level = Var(self.GENERICCAES, m.TIMESTEPS,
                             within=NonNegativeReals)

        # TES: Energy inflow
        self.tes_e_in = Var(self.GENERICCAES, m.TIMESTEPS,
                            within=NonNegativeReals)

        # TES: Energy outflow
        self.tes_e_out = Var(self.GENERICCAES, m.TIMESTEPS,
                             within=NonNegativeReals)

        # Spot market: Positive capacity
        self.exp_p_spot = Var(self.GENERICCAES, m.TIMESTEPS,
                              within=NonNegativeReals)

        # Spot market: Negative capacity
        self.cmp_p_spot = Var(self.GENERICCAES, m.TIMESTEPS,
                              within=NonNegativeReals)

        # Compression: Capacity on markets
        def cmp_p_constr_rule(block, n, t):
            expr = 0
            expr += -self.cmp_p[n, t]
            expr += m.flow[list(n.electrical_input.keys())[0], n, t]
            return expr == 0
        self.cmp_p_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_p_constr_rule)

        # Compression: Max. capacity depending on cavern filling level
        def cmp_p_max_constr_rule(block, n, t):
            if t != 0:
                return (self.cmp_p_max[n, t] ==
                        n.params['cmp_p_max_m'] * self.cav_level[n, t-1] +
                        n.params['cmp_p_max_b'])
            else:
                return (self.cmp_p_max[n, t] == n.params['cmp_p_max_b'])
        self.cmp_p_max_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_p_max_constr_rule)

        def cmp_p_max_area_constr_rule(block, n, t):
            return (self.cmp_p[n, t] <= self.cmp_p_max[n, t])
        self.cmp_p_max_area_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_p_max_area_constr_rule)

        # Compression: Status of operation (on/off)
        def cmp_st_p_min_constr_rule(block, n, t):
            return (
                self.cmp_p[n, t] >= n.params['cmp_p_min'] * self.cmp_st[n, t])
        self.cmp_st_p_min_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_st_p_min_constr_rule)

        def cmp_st_p_max_constr_rule(block, n, t):
            return (self.cmp_p[n, t] <=
                    (n.params['cmp_p_max_m'] * n.params['cav_level_max'] +
                    n.params['cmp_p_max_b']) * self.cmp_st[n, t])
        self.cmp_st_p_max_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_st_p_max_constr_rule)

        # (7) Compression: Heat flow out
        def cmp_q_out_constr_rule(block, n, t):
            return (self.cmp_q_out_sum[n, t] ==
                    n.params['cmp_q_out_m'] * self.cmp_p[n, t] +
                    n.params['cmp_q_out_b'] * self.cmp_st[n, t])
        self.cmp_q_out_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_q_out_constr_rule)

        #  (8) Compression: Definition of single heat flows
        def cmp_q_out_sum_constr_rule(block, n, t):
            return (self.cmp_q_out_sum[n, t] == self.cmp_q_waste[n, t] +
                    self.tes_e_in[n, t])
        self.cmp_q_out_sum_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_q_out_sum_constr_rule)

        # (9) Compression: Heat flow out ratio
        def cmp_q_out_shr_constr_rule(block, n, t):
            return (self.cmp_q_waste[n, t] * n.params['cmp_q_tes_share'] ==
                    self.tes_e_in[n, t] * (1 - n.params['cmp_q_tes_share']))
        self.cmp_q_out_shr_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cmp_q_out_shr_constr_rule)

        # (10) Expansion: Capacity on markets
        def exp_p_constr_rule(block, n, t):
            expr = 0
            expr += -self.exp_p[n, t]
            expr += m.flow[n, list(n.electrical_output.keys())[0], t]
            return expr == 0
        self.exp_p_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_p_constr_rule)

        # (11-12) Expansion: Max. capacity depending on cavern filling level
        def exp_p_max_constr_rule(block, n, t):
            if t != 0:
                return (self.exp_p_max[n, t] ==
                        n.params['exp_p_max_m'] * self.cav_level[n, t-1] +
                        n.params['exp_p_max_b'])
            else:
                return (self.exp_p_max[n, t] == n.params['exp_p_max_b'])
        self.exp_p_max_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_p_max_constr_rule)

        # (13)
        def exp_p_max_area_constr_rule(block, n, t):
            return (self.exp_p[n, t] <= self.exp_p_max[n, t])
        self.exp_p_max_area_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_p_max_area_constr_rule)

        # (14) Expansion: Status of operation (on/off)
        def exp_st_p_min_constr_rule(block, n, t):
            return (
                self.exp_p[n, t] >= n.params['exp_p_min'] * self.exp_st[n, t])
        self.exp_st_p_min_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_st_p_min_constr_rule)

        # (15)
        def exp_st_p_max_constr_rule(block, n, t):
            return (self.exp_p[n, t] <=
                    (n.params['exp_p_max_m'] * n.params['cav_level_max'] +
                     n.params['exp_p_max_b']) * self.exp_st[n, t])
        self.exp_st_p_max_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_st_p_max_constr_rule)

        # (16) Expansion: Heat flow in
        def exp_q_in_constr_rule(block, n, t):
            return (self.exp_q_in_sum[n, t] ==
                    n.params['exp_q_in_m'] * self.exp_p[n, t] +
                    n.params['exp_q_in_b'] * self.exp_st[n, t])
        self.exp_q_in_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_q_in_constr_rule)

        # (17) Expansion: Fuel allocation
        def exp_q_fuel_constr_rule(block, n, t):
            expr = 0
            expr += -self.exp_q_fuel_in[n, t]
            expr += m.flow[list(n.fuel_input.keys())[0], n, t]
            return expr == 0
        self.exp_q_fuel_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_q_fuel_constr_rule)

        # (18) Expansion: Definition of single heat flows
        def exp_q_in_sum_constr_rule(block, n, t):
            return (self.exp_q_in_sum[n, t] == self.exp_q_fuel_in[n, t] +
                    self.tes_e_out[n, t] + self.exp_q_add_in[n, t])
        self.exp_q_in_sum_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_q_in_sum_constr_rule)

        # (19) Expansion: Heat flow in ratio
        def exp_q_in_shr_constr_rule(block, n, t):
            return (n.params['exp_q_tes_share'] * self.exp_q_fuel_in[n, t] ==
                    (1 - n.params['exp_q_tes_share']) *
                    (self.exp_q_add_in[n, t] + self.tes_e_out[n, t]))
        self.exp_q_in_shr_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=exp_q_in_shr_constr_rule)

        # (20) Cavern: Energy inflow
        def cav_e_in_constr_rule(block, n, t):
            return (self.cav_e_in[n, t] ==
                    n.params['cav_e_in_m'] * self.cmp_p[n, t] +
                    n.params['cav_e_in_b'])
        self.cav_e_in_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cav_e_in_constr_rule)

        # (21) Cavern: Energy outflow
        def cav_e_out_constr_rule(block, n, t):
            return (self.cav_e_out[n, t] ==
                    n.params['cav_e_out_m'] * self.exp_p[n, t] +
                    n.params['cav_e_out_b'])
        self.cav_e_out_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cav_e_out_constr_rule)

        # (22-23) Cavern: Storage balance
        def cav_eta_constr_rule(block, n, t):
            if t != 0:
                return (n.params['cav_eta_temp'] * self.cav_level[n, t] ==
                        self.cav_level[n, t-1] + m.timeincrement[t] *
                        (self.cav_e_in[n, t] - self.cav_e_out[n, t]))
            else:
                return (n.params['cav_eta_temp'] * self.cav_level[n, t] ==
                        m.timeincrement[t] *
                        (self.cav_e_in[n, t] - self.cav_e_out[n, t]))
        self.cav_eta_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cav_eta_constr_rule)

        # (24) Cavern: Upper bound
        def cav_ub_constr_rule(block, n, t):
            return (self.cav_level[n, t] <= n.params['cav_level_max'])
        self.cav_ub_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=cav_ub_constr_rule)

        # (25-26) TES: Storage balance
        def tes_eta_constr_rule(block, n, t):
            if t != 0:
                return (self.tes_level[n, t] ==
                        self.tes_level[n, t-1] + m.timeincrement[t] *
                        (self.tes_e_in[n, t] - self.tes_e_out[n, t]))
            else:
                return (self.tes_level[n, t] ==
                        m.timeincrement[t] *
                        (self.tes_e_in[n, t] - self.tes_e_out[n, t]))
        self.tes_eta_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=tes_eta_constr_rule)

        # (27) TES: Upper bound
        def tes_ub_constr_rule(block, n, t):
            return (self.tes_level[n, t] <= n.params['tes_level_max'])
        self.tes_ub_constr = Constraint(
            self.GENERICCAES, m.TIMESTEPS, rule=tes_ub_constr_rule)
