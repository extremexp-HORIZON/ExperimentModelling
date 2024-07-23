package eu.extremexp.dms;

public interface DesignModelStorage {

    public Object get(String name, DesignModelStorageType kind) ;

    public void put(String name, Object element, DesignModelStorageType kind) ;
}
