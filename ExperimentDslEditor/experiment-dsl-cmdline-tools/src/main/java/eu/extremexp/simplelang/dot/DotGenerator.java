package eu.extremexp.simplelang.dot;

import eu.extremexp.simplelang.elements.*;

import java.util.*;


public class DotGenerator {
    private static final String WORKFLOW_BEGINNING = "node [shape=box];\nrankdir=TB;";
    private static final String GROUP_BEGINNING = "cluster=true;\nrank=same;";
    private static final String START_NODE = "START [shape=circle,style=filled,color=black,label=\"\"];";
    private static final String END_NODE = "END [shape=doublecircle,style=filled,color=black,label=\"\"];";
    private static final String PARALLEL_NODE = " [shape=diamond,label=<<FONT POINT-SIZE=\"20\">+</FONT>>];";
    private static final String EXCLUSIVE_NODE = " [shape=diamond,label=<<FONT POINT-SIZE=\"20\">&nbsp;</FONT>>];";

    final private Workflow workflow;
    final private List<String> dot;

    private static boolean hasStart = false;
    private static boolean hasEnd = false;
    private final Set<String> operators = new HashSet<>();
    private final Set<String> dataNodes = new HashSet<>();

    public DotGenerator(Workflow workflow) {
        this.workflow = workflow;
        this.dot = generate();
    }

    public List<String> getDot() {
        return dot;
    }

    private static String edgeToString(Edge edge) {
        return String.join(" -> ", edge.nodes);
    }

    /*
    private static String vectorAttributeToString(Node node, String attributeName) {
        var value = node.attributes.stream().filter(x -> x.key.equals(attributeName)).map(x -> x.value).findFirst();
        if (value.isEmpty()) {
            return "";
        } else {
            switch (value.get()) {
                case VectorValue vectorValue -> {
                    return "<BR/>"+ String.join("<BR/>", vectorValue.value);
                }
                case ScalarValue scalarValue -> {
                    return "<BR/>" + scalarValue.value;
                }
            }
        }
    }*/

    private static Pair<String, String> subworkflowLabel(Node node) {
        Optional<String> subworkflow = node.attributes.stream().filter(x -> x.key().equals("subworkflow")).map(x -> x.element()).findFirst();
        return subworkflow.map(value -> new Pair<>("<BR/><FONT POINT-SIZE=\"30\">&#x229E;</FONT>", value)).orElseGet(() -> new Pair<>("", ""));
    }

    /**
     * Generates the dot specification corresponding to the workflow.
     * @return the dot specification as a list of lines
     */
    private List<String> generate() {
        List<String> ret = new ArrayList<>();
        ret.add("digraph " + workflow.id + "{");
        ret.add(WORKFLOW_BEGINNING);

        workflow.elements.forEach(element -> processElement(element, ret));

        ret.add("}");
        return ret;
    }

    private void processElement(Element element, List<String> ret) {
        switch (element) {
            case Edge edge -> {
                for (String node: edge.nodes) {
                    if ("START".equals(node) && !hasStart) {
                        ret.add(START_NODE);
                        hasStart = true;
                    } else if ("END".equals(node) && !hasEnd) {
                        ret.add(END_NODE);
                        hasEnd = true;
                    } else if (node.startsWith("PARALLEL_") && !operators.contains(node)) {
                        ret.add(node + PARALLEL_NODE);
                        operators.add(node);
                    } else if (node.startsWith("EXCLUSIVE_") && !operators.contains(node)) {
                        ret.add(node + EXCLUSIVE_NODE);
                        operators.add(node);
                    }
                }
                String line = edgeToString(edge);
                switch (edge) {
                    case RegularEdge ignored -> {}
                    case ConditionalEdge ignored -> {
                        String condition = "";
                        for (var attr: edge.attributes) {
                            if (attr.key().equals("condition")) {
                                condition = attr.element();
                            }
                        }
                        if (!condition.isEmpty()) {
                            condition = ",label="+ condition;
                        }
                        line += "[arrowtail=ediamond,dir=both"  + condition + "];";
                    }
                    case DataEdge ignored -> { line += " [ style=dashed ];"; }
                }
                ret.add(line);
            }
            case Node node -> {
                if (node == Node.IGNORED) {
                    return;
                }
                StringBuilder line = new StringBuilder();
                if (node.isData) {
                    //String id = "<<B>" + node.id + "</B>" + vectorAttributeToString(node, "data") + ">";
                    String id = "<<B>" + node.id + "</B>>";
                    line.append(node.id).append(" [label=").append(id).append(",style=dashed]");
                    dataNodes.add(node.id);
                } else { // regular node
                    var subworkflow = subworkflowLabel(node);
                    //String id = "<<B>" + node.id + "</B>" + vectorAttributeToString(node, "parameters") + subworkflow.key + ">";
                    String id = "<<B>" + node.id + "</B>" + subworkflow.key + ">";
                    line.append(node.id).append(" [label=").append(id);

                    line.append("]");
                }
                line.append(";");
                ret.add(line.toString());
            }
            case Group group -> {
                ret.add("subgraph " + "{");
                ret.add(GROUP_BEGINNING);
                ret.add("label=" + group.id + ";");
                group.elements.forEach(el -> processElement(el, ret));
                ret.add("}");
            }
            case Workflow ignored -> {
                throw new RuntimeException("Unsupported workflow in workflow");
            }
        }
    }
}
