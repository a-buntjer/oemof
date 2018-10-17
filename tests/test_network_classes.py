# -*- coding: utf-8 -

"""Test the created constraints against approved constraints.

This file is part of project oemof (github.com/oemof/oemof). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location oemof/tests/test_network_classes.py

SPDX-License-Identifier: GPL-3.0-or-later
"""

from traceback import format_exception_only as feo
from nose.tools import assert_raises, eq_, ok_
import warnings

from oemof.energy_system import EnergySystem as ES
from oemof.network import Bus, Edge, Node, Transformer
from oemof import graph
from oemof.solph import Model


class Node_Tests:

    def setup(self):
        self.energysystem = ES()
        Node.registry = self.energysystem

    def test_that_attributes_cannot_be_added(self):
        node = Node()
        with assert_raises(AttributeError):
            node.foo = "bar"

    def test_symmetric_input_output_assignment(self):
        n1 = Node(label="<N1>")

        n2 = Node(label="<From N1>", inputs=[n1])
        ok_(n1 in n2.inputs,
            "{0} not in {1}.inputs, ".format(n1, n2) +
            "although it should be by construction.")
        ok_(n2 in n1.outputs,
            "{0} in {1}.inputs but {1} not in {0}.outputs.".format(n1, n2))

        n3 = Node(label="<To N1>", outputs=[n1])
        ok_(n1 in n3.outputs,
            "{0} not in {1}.outputs, ".format(n1, n3) +
            "although it should be by construction.")
        ok_(n3 in n1.inputs,
            "{0} in {1}.outputs but {1} not in {0}.inputs.".format(n1, n3))

    def test_accessing_outputs_of_a_node_without_output_flows(self):
        n = Node()
        exception = None
        try:
            outputs = n.outputs
        except Exception as e:
            exception = e
        ok_(exception is None,
            "\n  Test accessing `outputs` on {} having no outputs.".format(n) +
            "\n  Got unexpected exception:\n" +
            "\n      {}".format(feo(type(exception), exception)[0]))
        ok_(len(outputs) == 0,
            "\n  Failure when testing `len(outputs)`." +
            "\n  Expected: 0." +
            "\n  Got     : {}".format(len(outputs)))

    def test_accessing_inputs_of_a_node_without_input_flows(self):
        n = Node()
        exception = None
        try:
            inputs = n.inputs
        except Exception as e:
            exception = e
        ok_(exception is None,
            "\n  Test accessing `inputs` on {} having no inputs.".format(n) +
            "\n  Got unexpected exception:\n" +
            "\n      {}".format(feo(type(exception), exception)[0]))
        ok_(len(inputs) == 0,
            "\n  Failure when testing `len(inputs)`." +
            "\n  Expected: 0." +
            "\n  Got     : {}".format(len(inputs)))

    def test_that_the_outputs_attribute_of_a_node_is_a_mapping(self):
        n = Node()
        exception = None
        try:
            values = n.outputs.values()
        except AttributeError as e:
            exception = e
        ok_(exception is None,
            "\n  Test accessing `outputs.values()`" +
            " on {} having no inputs.".format(n) +
            "\n  Got unexpected exception:\n" +
            "\n      {}".format(feo(type(exception), exception)[0]))

    def test_that_nodes_do_not_get_undead_flows(self):
        """ Newly created nodes should only have flows assigned to them.

        A new node `n`, which re-used a previously used label `l`, retained the
        flows of those nodes which where labeled `l` before `n`. This incorrect
        behaviour is a problem if somebody wants to use different nodes with
        the same label in multiple energy systems. While this feature currently
        isn't used, it also lead to weird behaviour when running tests.

        This test ensures that new nodes only have those flows which are
        assigned to them on construction.
        """
        # We don't want a registry as we are re-using a label on purpose.
        # Having a registry would just throw and error generated by the DEFAULT
        # grouping in this case.
        Node.registry = None
        flow = object()
        old = Node(label="A reused label")
        bus = Bus(label="bus", inputs={old: flow})
        eq_(bus.inputs[old].flow, flow,
            ("\n  Expected: {}" +
             "\n  Got     : {} instead").format(flow, bus.inputs[old].flow))
        eq_(old.outputs[bus].flow, flow,
            ("\n  Expected: {}" +
             "\n  Got     : {} instead").format(flow, old.outputs[bus].flow))
        new = Node(label="A reused label")
        eq_(new.outputs, {},
            ("\n  Expected an empty dictionary of outputs." +
             "\n  Got: {} instead").format(new.outputs))

    def test_modifying_outputs_after_construction(self):
        """ One should be able to add and delete outputs of a node.
        """
        node = Node("N1")
        bus = Node("N2")
        flow = "flow"
        eq_(node.outputs, {},
            ("\n  Expected an empty dictionary of outputs." +
             "\n  Got: {} (== {}) instead").format(
                node.outputs,
                dict(node.outputs)))
        node.outputs[bus] = flow
        eq_(node.outputs, {bus: flow},
            ("\n  Expected {} as `node.outputs`." +
             "\n  Got    : {} (== {}) instead").format(
                {bus: flow}, node.outputs,dict(node.outputs)))
        eq_(node.outputs[bus], flow,
            ("\n  Expected {} as `node.outputs[bus]`." +
             "\n  Got    : {} instead").format(flow, node.outputs[bus]))
        del node.outputs[bus]
        eq_(node.outputs, {},
            ("\n  Expected an empty dictionary of outputs." +
             "\n  Got: {} (== {}) instead").format(
                node.outputs,
                dict(node.outputs)))

    def test_modifying_inputs_after_construction(self):
        """ One should be able to add and delete inputs of a node.
        """
        node = Node("N1")
        bus = Bus("N2")
        flow = "flow"

        eq_(node.inputs, {},
            ("\n  Expected an empty dictionary of inputs." +
             "\n  Got: {} (== {}) instead").format(
                node.inputs,
                dict(node.inputs)))
        node.inputs[bus] = flow
        eq_(node.inputs, {bus: flow},
            ("\n  Expected {} as `node.inputs`." +
             "\n  Got    : {} (== {}) instead").format(
                {bus: flow}, node.inputs,dict(node.inputs)))
        eq_(node.inputs[bus], flow,
            ("\n  Expected {} as `node.inputs[bus]`." +
             "\n  Got    : {} instead").format(flow, node.inputs[bus]))
        del node.inputs[bus]
        eq_(node.inputs, {},
            ("\n  Expected an empty dictionary of inputs." +
             "\n  Got: {} (== {}) instead").format(
                node.inputs,
                dict(node.inputs)))

    def test_output_input_symmetry_after_modification(self):
        n1 = Node("N1")
        n2 = Node("N2")
        flow = "flow"

        n1.outputs[n2] = flow
        eq_(n2.inputs, {n1: flow})

    def test_input_output_symmetry_after_modification(self):
        n1 = Node("N1")
        n2 = Node("N2")
        flow = "flow"

        n1.inputs[n2] = flow
        eq_(n2.outputs, {n1: flow})

    def test_updating_inputs(self):
        n1 = Node("N1")
        n2 = Node("N2")
        n1n2 = "n1n2"

        n2.inputs.update({n1: n1n2})
        eq_(n2.inputs, {n1: n1n2})
        eq_(n2.inputs[n1], n1n2)
        eq_(n1.outputs, {n2: n1n2})
        eq_(n1.outputs[n2], n1n2)

    def test_updating_outputs(self):
        n1 = Node("N1")
        n2 = Node("N2")
        n1n2 = "n1n2"

        n1.outputs.update({n2: n1n2})
        eq_(n2.inputs, {n1: n1n2})
        eq_(n2.inputs[n1], n1n2)
        eq_(n1.outputs, {n2: n1n2})
        eq_(n1.outputs[n2], n1n2)


class Edge_Tests:
    def test_edge_construction_side_effects(self):
        """ Constructing an `Edge` should affect it's input/output `Node`s.

        When constructing an `Edge`, the `inputs` and `outputs` of its output
        and input `Node`s should be set appropriately.
        """
        source = Node(label='source')
        target = Node(label='target')
        edge = Edge(input=source, output=target)
        ok_(target in source.outputs,
            "{} not in {} after constructing {}."
            .format(target, source.outputs, edge))
        ok_(source in target.inputs,
            "{} not in {} after constructing {}."
            .format(source, target.outputs, edge))

    def test_label_as_positional_argument(self):
        o = object()
        n = Node(o)
        ok_(n.label is o, (
                "Setting `label` as positional parameter argument failed."
                "\n  Expected: {!r}"
                "\n  Got     : {!r}")
                .format(o, n.label))


class EnergySystem_Nodes_Integration_Tests:

    def setup(self):
        self.es = ES()
        Node.registry = self.es

    def test_node_registration(self):
        eq_(Node.registry, self.es)
        b1 = Bus(label='<B1>')
        eq_(self.es.entities[0], b1)
        b2 = Bus(label='<B2>')
        eq_(self.es.entities[1], b2)
        t1 = Transformer(label='<TF1>', inputs=[b1], outputs=[b2])
        ok_(t1 in self.es.entities)


def test_depreciated_graph_call():
    es = ES()
    om = Model(energysystem=es)
    warnings.filterwarnings('ignore', category=FutureWarning)
    graph.create_nx_graph(optimization_model=om)
