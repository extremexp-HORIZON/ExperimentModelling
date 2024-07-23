package eu.extremexp.simplelang.elements;

import java.util.List;

public final class Workflow extends Element {
    final public String packageName;
    final public String id;
    final public List<Element> elements;
    final public List<Pair<String, Value>> attributes;

    public Workflow(String packageName, String id, List<Element> elements, List<Pair<String, Value>> attributes) {
        this.packageName = packageName;
        this.id = id;
        this.elements = elements == null ? List.of(): elements;
        this.attributes = attributes == null ? List.of() : attributes;
    }
}
