/*
 * generated by Xtext 2.32.0
 */
package eu.extremexp.dsl.ide

import com.google.inject.Guice
import eu.extremexp.dsl.XDSLRuntimeModule
import eu.extremexp.dsl.XDSLStandaloneSetup
import org.eclipse.xtext.util.Modules2

/**
 * Initialization support for running Xtext languages as language servers.
 */
class XDSLIdeSetup extends XDSLStandaloneSetup {

	override createInjector() {
		Guice.createInjector(Modules2.mixin(new XDSLRuntimeModule, new XDSLIdeModule))
	}
	
}
