# -*- coding: utf-8 -*-

"""Optional classes to be added to a network class.
This file is part of project oemof (github.com/oemof/oemof). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location oemof/oemof/solph/options.py

SPDX-License-Identifier: GPL-3.0-or-later
"""

from oemof.solph.plumbing import sequence


class Investment:
    """
    Parameters
    ----------
    maximum : float
        Maximum of the additional invested capacity
    minimum : float
        Minimum of the additional invested capacity
    ep_costs : float
        Equivalent periodical costs for the investment, if period is one
        year these costs are equal to the equivalent annual costs.
    existing : float
        Existing / installed capacity. The invested capacity is added on top
        of this value.

    """
    def __init__(self, maximum=float('+inf'), minimum=0, ep_costs=0,
                 existing=0):
        self.maximum = maximum
        self.minimum = minimum
        self.ep_costs = ep_costs
        self.existing = existing


class NonConvex:
    """
    Parameters
    ----------
    startup_costs : numeric (sequence or scalar)
        Costs associated with a start of the flow (representing a unit).
    shutdown_costs : numeric (sequence or scalar)
        Costs associated with the shutdown of the flow (representing a unit).
    activity_costs : numeric (sequence or scalar)
        Costs associated with the active operation of the flow, independently
        from the actual output.
    minimum_uptime : numeric (1 or positive integer)
        Minimum time that a flow must be greater then its minimum flow after
        startup. Be aware that minimum up and downtimes can contradict each
        other and may lead to infeasible problems.
    minimum_downtime : numeric (1 or positive integer)
        Minimum time a flow is forced to zero after shutting down.
        Be aware that minimum up and downtimes can contradict each
        other and may to infeasible problems.
    maximum_startups : numeric (0 or positive integer)
        Maximum number of start-ups.
    maximum_shutdowns : numeric (0 or positive integer)
        Maximum number of shutdowns.
    initial_status : numeric (0 or 1)
        Integer value indicating the status of the flow in the first time step
        (0 = off, 1 = on). For minimum up and downtimes, the initial status
        is set for the respective values in the edge regions e.g. if a
        minimum uptime of four timesteps is defined, the initial status is
        fixed for the four first and last timesteps of the optimization period.
        If both, up and downtimes are defined, the initial status is set for
        the maximum of both e.g. for six timesteps if a minimum downtime of
        six timesteps is defined in addition to a four timestep minimum uptime.
    """
    def __init__(self, **kwargs):
        scalars = ['minimum_uptime', 'minimum_downtime', 'initial_status',
                   'maximum_startups', 'maximum_shutdowns']
        sequences = ['startup_costs', 'shutdown_costs', 'activity_costs']
        defaults = {'initial_status': 0}

        for attribute in set(scalars + sequences + list(kwargs)):
            value = kwargs.get(attribute, defaults.get(attribute))
            setattr(self, attribute,
                    sequence(value) if attribute in sequences else value)

        self._max_up_down = None

    def _calculate_max_up_down(self):
        """
        Calculate maximum of up and downtime for direct usage in constraints.

        The maximum of both is used to set the initial status for this
        number of timesteps within the edge regions.
        """
        if self.minimum_uptime is not None and self.minimum_downtime is None:
            max_up_down = self.minimum_uptime
        elif self.minimum_uptime is None and self.minimum_downtime is not None:
            max_up_down = self.minimum_downtime
        else:
            max_up_down = max(self.minimum_uptime, self.minimum_downtime)

        self._max_up_down = max_up_down

    @property
    def max_up_down(self):
        """Compute or return the _max_up_down attribute."""
        if self._max_up_down is None:
            self._calculate_max_up_down()

        return self._max_up_down


class RollingHorizon:
    """
    Parameters
    ----------
    startup_costs : numeric (sequence or scalar)
        Costs associated with a start of the flow (representing a unit).
    shutdown_costs : numeric (sequence or scalar)
        Costs associated with the shutdown of the flow (representing a unit).
    activity_costs : numeric (sequence or scalar)
        Costs associated with the active operation of the flow, independently
        from the actual output.
    minimum_uptime : numeric (1 or positive integer)
        Minimum time that a flow must be greater then its minimum flow after
        startup. Be aware that minimum up and downtimes can contradict each
        other and may lead to infeasible problems.
    minimum_downtime : numeric (1 or positive integer)
        Minimum time a flow is forced to zero after shutting down.
        Be aware that minimum up and downtimes can contradict each
        other and may to infeasible problems.
    maximum_startups : numeric (0 or positive integer)
        Maximum number of start-ups.
    maximum_shutdowns : numeric (0 or positive integer)
        Maximum number of shutdowns.
    initial_status : numeric (0 or 1)
        Integer value indicating the status of the flow in the first time step
        (0 = off, 1 = on). For minimum up and downtimes, the initial status
        is set for the respective values in the edge regions e.g. if a
        minimum uptime of four timesteps is defined, the initial status is
        fixed for the four first and last timesteps of the optimization period.
        If both, up and downtimes are defined, the initial status is set for
        the maximum of both e.g. for six timesteps if a minimum downtime of
        six timesteps is defined in addition to a four timestep minimum uptime.
    """
    def __init__(self, **kwargs):
        scalars = ['minimum_uptime', 'minimum_downtime', 't_start_cold',
                   't_start_warm', 't_start_hot', 'maximum_startups',
                   'maximum_shutdowns', 'ramp_limit_up', 'ramp_limit_down',
                   'ramp_limit_start_up', 'ramp_limit_shut_down', 'tau',
                   't_warm', 't_cold', 'T_int', 'flow_min_last', 'T']
        sequences = ['cold_start_costs', 'warm_start_costs', 'hot_start_costs',
                     'shutdown_costs']
        defaults = {'minimum_uptime': 0, 'minimum_downtime': 1,
                    't_start_cold': 3, 't_start_warm': 2, 't_start_hot': 1,
                    'tau': 1, 't_warm': 8, 't_cold': 48, 'T_int': 0, 'T':0}

        for attribute in set(scalars + sequences + list(kwargs)):
            value = kwargs.get(attribute, defaults.get(attribute))
            setattr(self, attribute,
                    sequence(value) if attribute in sequences else value)
        self.optimized_status = [0 for x in range(self.T_int+1)]
        self.optimized_flow = [0 for x in range(self.T_int+1)]
        self.T_offl_hs = [0 for x in range(self.T_int+1)]
        self.T_offl_ws = [0 for x in range(self.T_int+1)]
        self.T_offl_cs = [0 for x in range(self.T_int+1)]
        self.T_wsc = [0 for x in range(self.T_int+1)]
        self.T_csc = [0 for x in range(self.T_int+1)]
        self.xi_ini_ws = [0 for x in range(self.T_int+1)]
        self.xi_ini_cs = [0 for x in range(self.T_int+1)]

    @property
    def initial_status(self):
        """Compute and return the initial_status attribute."""
        if self.optimized_status is None:
            return 0
        else:
            return self.optimized_status[self.T-1]

    @property
    def T_offl_min_hs(self):
        """Compute and return the _T_offl_min_hs attribute."""
        self._T_offl_min_hs = int((
                self.minimum_downtime + self.t_start_hot)/self.tau)
        return self._T_offl_min_hs

    @property
    def T_offl_min_ws(self):
        """Compute and return the _T_offl_min_hs attribute."""
        self._T_offl_min_ws = int((self.t_warm + self.t_start_warm)/self.tau)
        return self._T_offl_min_ws

    @property
    def T_offl_min_cs(self):
        """Compute and return the _T_offl_min_hs attribute."""
        self._T_offl_min_cs = int((self.t_cold + self.t_start_cold)/self.tau)
        return self._T_offl_min_cs

    @property
    def T_offl_th_ws(self):
        """Compute and return the _T_offl_min_hs attribute."""
        self._T_offl_th_ws = int((self.t_warm + self.t_start_hot)/self.tau)
        return self._T_offl_th_ws

    @property
    def T_offl_th_cs(self):
        """Compute and return the _T_offl_min_hs attribute."""
        self._T_offl_th_cs = int((self.t_cold + self.t_start_warm)/self.tau)
        return self._T_offl_th_cs

    @property
    def T_up_min(self):
        """Compute and return the _T_offl_min_hs attribute."""
        self._T_up_min = (self.minimum_uptime)/self.tau
        return self._T_up_min

    def _calculate_helper_variables(self):
        """
        Calculate maximum of up and downtime for direct usage in constraints.

        The maximum of both is used to set the initial status for this
        number of timesteps within the edge regions.
        """
        helper_variables = {'sum_start_ini': 0, 'T_ini': 0, 'Z_ini_ws': 0,
                            'Z_ini_cs': 0, 'T_offl_possible_ws': 0,
                            'T_offl_possible_cs': 0}
        if self.T > 0:
            last_state = self.optimized_status[self.T-1]
            for status in reversed(self.optimized_status[:self.T]):
                if (status == last_state):
                    helper_variables['sum_start_ini'] += 1
                    last_state = status
                else:
                    break
            # T_ini
            if (self.initial_status == 0) and\
                    (helper_variables['sum_start_ini'] <
                     self.T_offl_min_hs):
                helper_variables['T_ini'] =\
                    min(self.T_int, self.T_offl_min_hs -
                        helper_variables['sum_start_ini'])
            elif (self.initial_status == 0) and\
                    (self.T_offl_min_ws >
                     helper_variables['sum_start_ini']) and\
                    (helper_variables['sum_start_ini'] >= self.T_offl_th_ws):
                helper_variables['T_ini'] =\
                    min(self.T_int, self.T_offl_min_ws -
                        helper_variables['sum_start_ini'])
            elif (self.initial_status == 0) and\
                    (self.T_offl_min_cs >
                     helper_variables['sum_start_ini']) and\
                    (helper_variables['sum_start_ini'] >= self.T_offl_th_cs):
                helper_variables['T_ini'] =\
                    min(self.T_int, self.T_offl_min_cs -
                        helper_variables['sum_start_ini'])
            elif (self.initial_status == 1) and\
                    (helper_variables['sum_start_ini'] <
                     self.T_up_min):
                helper_variables['T_ini'] =\
                    min(self.T_int, self.T_up_min -
                        helper_variables['sum_start_ini'])
            else:
                0
            # Z ini warm start
            if (self.initial_status == 0) and\
                    (helper_variables['sum_start_ini'] <
                     self.T_offl_th_ws) and\
                    (self.T_int > self.T_offl_th_ws -
                     helper_variables['sum_start_ini']):
                helper_variables['Z_ini_ws'] = 1
            else:
                helper_variables['Z_ini_ws'] = 0
            # Z ini cold start
            if (self.initial_status == 0) and\
                    (helper_variables['sum_start_ini'] <
                     self.T_offl_th_cs) and\
                    (self.T_int > self.T_offl_th_cs -
                     helper_variables['sum_start_ini']):
                helper_variables['Z_ini_cs'] = 1
            else:
                helper_variables['Z_ini_cs'] = 0
            # T offl possible warm start
            if self.T_int >= self.T_offl_min_ws -\
                    helper_variables['sum_start_ini']:
                helper_variables['T_offl_possible_ws'] =\
                    self.T_offl_min_ws -\
                    self.T_offl_th_ws-1
            else:
                helper_variables['T_offl_possible_ws'] =\
                    self.T_int-1-self.T_offl_min_ws +\
                    helper_variables['sum_start_ini']
            # T offl possible cold start
            if self.T_int >= self.T_offl_min_cs -\
                    helper_variables['sum_start_ini']:
                helper_variables['T_offl_possible_cs'] =\
                    self.T_offl_min_cs -\
                    self.T_offl_th_cs-1
            else:
                helper_variables['T_offl_possible_cs'] =\
                    self.T_int-1-self.T_offl_min_cs +\
                    helper_variables['sum_start_ini']
        self._helper_variables = helper_variables

    @property
    def helper_variables(self):
        """Compute and return the _helper_variables attribute."""
        self._calculate_helper_variables()
        return self._helper_variables
