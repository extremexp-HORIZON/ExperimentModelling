package eu.extremexp.simplelang.elements;

import java.util.Collections;
import java.util.List;

public final class Node extends Element {

    public static final Node IGNORED = new Node("<IGNORED>", false, List.of());

    public final String id;
    public final List<ConfigElement> attributes;
    public final boolean isData;

    public Node(String id, boolean isData, List<ConfigElement> attributes) {
        this.id = id;
        this.isData = isData;
        this.attributes = Collections.unmodifiableList(attributes);
    }
}
