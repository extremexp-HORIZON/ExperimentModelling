package eu.extremexp.dms;

import com.google.inject.Injector;
import eu.extremexp.dsl.LanguageStandaloneSetup;
import eu.extremexp.dsl.language.LanguagePackage;
import org.eclipse.xtext.resource.XtextResource;
import org.eclipse.xtext.resource.XtextResourceSet;
public abstract class  AbstractDMS implements DesignModelStorage{

    protected  XtextResourceSet resourceSet;
    public AbstractDMS () {
        LanguagePackage.eINSTANCE.eClass();
        Injector injector = new LanguageStandaloneSetup().createInjectorAndDoEMFRegistration();
        resourceSet =  injector.getInstance(XtextResourceSet.class);
        resourceSet.addLoadOption(XtextResource.OPTION_RESOLVE_ALL, true);
    }
}
