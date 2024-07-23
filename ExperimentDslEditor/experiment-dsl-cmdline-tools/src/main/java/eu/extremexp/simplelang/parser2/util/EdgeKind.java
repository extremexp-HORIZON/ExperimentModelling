package eu.extremexp.simplelang.parser2.util;

import java.util.List;
import eu.extremexp.simplelang.elements.*;

public enum EdgeKind {
    REGULAR, DATA, CONDITIONAL;

    public Edge create(List<String> nodes, List<ConfigElement> attributes) {
        return switch (this) {
            case REGULAR -> new RegularEdge(nodes, attributes);
            case CONDITIONAL -> new ConditionalEdge(nodes, attributes);
            case DATA -> new DataEdge(nodes, attributes);
        };
    }
}
