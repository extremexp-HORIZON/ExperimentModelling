package eu.extremexp.simplelang.elements;

import java.util.Collections;
import java.util.List;

public sealed abstract class Edge extends Element permits ConditionalEdge, RegularEdge, DataEdge {
    public final List<ConfigElement> attributes;
    public final List<String> nodes;

    public Edge(List<String> nodes, List<ConfigElement> attributes) {
        this.nodes = nodes;
        this.attributes = Collections.unmodifiableList(attributes);
    }
}
