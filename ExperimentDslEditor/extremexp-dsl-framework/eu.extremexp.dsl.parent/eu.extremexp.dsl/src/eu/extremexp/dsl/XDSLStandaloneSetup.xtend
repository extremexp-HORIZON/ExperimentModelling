/*
 * generated by Xtext 2.32.0
 */
package eu.extremexp.dsl


/**
 * Initialization support for running Xtext languages without Equinox extension registry.
 */
class XDSLStandaloneSetup extends XDSLStandaloneSetupGenerated {

	def static void doSetup() {
		new XDSLStandaloneSetup().createInjectorAndDoEMFRegistration()
	}
}