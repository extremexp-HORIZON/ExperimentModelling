package eu.extremexp.simplelang.elements;

import java.util.Collections;
import java.util.List;

public final class VectorValue extends Value {
    public final List<String> value;

    public VectorValue(List<String> value) {
        this.value = Collections.unmodifiableList(value);
    }

    public VectorValue(String... values) {
        this.value = List.of(values);
    }
}
