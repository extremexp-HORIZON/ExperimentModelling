package eu.extremexp.simplelang.elements;

public record ConfigElement(String key, String element, Value value) {
     public static ConfigElement createImplementation(String element) {
         return new ConfigElement("implementation", element, null);
     }

     public static ConfigElement createSchema(String element) {
         return new ConfigElement("schema", element, null);
     }

    public static ConfigElement createCondition(String element) {
        return new ConfigElement("condition", element, null);
    }

    public static ConfigElement createParam(String element, Value value) {
        return new ConfigElement("schema", element, value);
    }
}
