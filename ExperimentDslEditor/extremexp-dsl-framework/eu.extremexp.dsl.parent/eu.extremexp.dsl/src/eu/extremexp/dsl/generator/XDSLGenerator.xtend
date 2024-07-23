/*
 * generated by Xtext 2.32.0
 */
package eu.extremexp.dsl.generator

import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.AbstractGenerator
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext
import eu.extremexp.dsl.xDSL.Namespace
import eu.extremexp.dsl.xDSL.Workflow

/**
 * Generates code from your model files on save.
 * 
 * See https://www.eclipse.org/Xtext/documentation/303_runtime_concepts.html#code-generation
 */
class XDSLGenerator extends AbstractGenerator {

	override void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		resource.allContents.filter(Workflow).forEach[
			e | 
			fsa.generateFile(e.name+'.json', converToJSON(e))
			fsa.generateFile(e.name+'.xml', convertToXML(e)
			)
		]
		
	}
	
	def convertToXML(Workflow wf) {
		'''
		<worklfow name=«wf.name»>
		«FOR task:wf.tasks»
			<Task name=«task.name»> 
			</Task>
		«ENDFOR»
		</workflow>
		'''
	}
	
	def converToJSON(Workflow wf) {
		'''
		{
			"name": "«wf.name»",
			"tasks": [
					«FOR tc:wf.taskConfigurations SEPARATOR  ','»
					{
						"name": "«tc.task.name»",
						"params" : [
						«FOR param:tc.params  SEPARATOR ',' »
							«IF param.otherParam === null»
							"«param.name»": «param.arbitrary.double»
							«ELSE»
							"«param.name»": «param.otherParam.name»
							«ENDIF»
						«ENDFOR»
						]
					}
					«ENDFOR»
				]
		}
		'''
	}
	
}