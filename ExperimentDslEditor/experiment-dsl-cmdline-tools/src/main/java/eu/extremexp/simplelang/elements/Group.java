package eu.extremexp.simplelang.elements;

import java.util.Collections;
import java.util.List;

public final class Group extends Element {
    public final String id;
    public final List<Element> elements;

    public Group(String id, List<Element> elements) {
        this.id = id;
        this.elements = Collections.unmodifiableList(elements);
    }
}
