# extremexp-metamodels

This project is for holding the meta-models of ExtremeXP, which are to be used to aligning the tool development in WP5. 

To try it out, you should download the [Eclipse Modeling Tools IDE](https://www.eclipse.org/downloads/packages/release/2023-12/r/eclipse-modeling-tools).

### Usage 

Once you import the project in Eclipse, you can navigate to the "model" folder and open the .ecore file to inspect it. 
It's a good idea to switch to the "Modeling" perspective of Eclipse, if not using this perspective already (Go to top menu -> Window -> Perspective -> Open Perspective)

For viewing and editing the .ecore model in a graphical editor, you can double-click on the .aird file and select the "workflow" representation under the Design category.

For generating the model (Java) code, open the .genmodel file (with the EMF Generator editor), right click on the first element in the genmodel and select Generate Model Code.   
Three Java packages will appear (if not already there) in the src-gen folder of this project. 