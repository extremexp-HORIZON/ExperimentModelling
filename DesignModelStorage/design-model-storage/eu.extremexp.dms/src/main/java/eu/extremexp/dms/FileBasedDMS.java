package eu.extremexp.dms;

import eu.extremexp.dsl.language.*;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

public class FileBasedDMS extends AbstractDMS {

    protected HashMap<String, HashMap<String, WorkflowInterface>> namespaces;

    @Override
    public Object get(String name, DesignModelStorageType kind) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void put(String name, Object element, DesignModelStorageType kind) {
        // TODO Auto-generated method stub

    }


    public FileBasedDMS () {
        namespaces = new HashMap<>();
    }

    private Model xxpFileToModel(String filepath) throws IOException{
        Resource resource = resourceSet.getResource(URI.createFileURI(filepath), true);
        return (Model) resource.getContents().get(0);
    }
    private static boolean isXXPFile(Path path, LinkOption... options) {
        return path.toString().toLowerCase().endsWith(".xxp");
    }

    private void showNamespaces(){
        namespaces.forEach(
                (namespaceName, stringWorkflowInterfaceHashMap) -> {
                    System.out.println("package: " + namespaceName);
                    stringWorkflowInterfaceHashMap.forEach(
                            (workflowName, workflowInterface) -> {
                                if (workflowInterface instanceof Workflow){
                                    System.out.println("\tWorkflow "+ workflowName);
                                    for (var task : ((Workflow) workflowInterface).getTasks()){
                                        System.out.println("\t\tTask: " + task.getName());
                                    }
                                    for (var data: ((Workflow) workflowInterface).getData()){
                                        System.out.println("\t\tData: " + data.getName());
                                    }
                                    for (var param: ((Workflow) workflowInterface).getParams()){
                                        System.out.println("\t\tParam: " + param.getName());
                                    }
                                }
                                else {
                                    //var parentWorkflow =  EcoreUtil2.resolve(((AssembledWorkflow) workflowInterface).getWorkflow(), workflowInterface.eResource().getResourceSet());
                                    System.out.println("\tAssembled Workflow: "+ workflowName + " (from " + ((AssembledWorkflow) workflowInterface).getWorkflow() + ")");
                                }

                            }

                    );
                }

        );
    }
    private void loadNamespaces (String directoryPath)  {
        try (Stream<Path> paths = Files.walk(Paths.get(directoryPath))) {
            List<Path> resolvedPaths = paths.filter(Files::isRegularFile).filter(FileBasedDMS::isXXPFile).toList();
            for (var path : resolvedPaths){
                Model model = xxpFileToModel(path.toString());
                String namespaceName = model.getNamespace().getName();
                WorkflowInterface wi = model.getNamespace().getWorkflows().get(0);

                if (wi == null){
                    continue;
                }

                if (!namespaces.containsKey(namespaceName)) {
                    namespaces.put(namespaceName, new HashMap<>());
                }
                var workflows = namespaces.get((namespaceName));
                workflows.putIfAbsent(wi.getName(), wi);

            }
        }
        catch (IOException _ioe) {
            System.out.println("path does not exist: " + _ioe.getLocalizedMessage());
        }
        catch (Exception ex) {
            System.out.println("Syntax error:" + ex.getMessage());
        }


    }

    private void runInteractiveMode(){
        String mainCommand = "";
        while (!mainCommand.equals("quit")){

            System.out.print(">> ");
            Scanner scanner = new Scanner(System.in);
            String commandLine = scanner.nextLine();
            String [] commandChain = commandLine.split(" ");

            if (commandChain.length == 0){
                continue;
            }

            mainCommand = commandChain[0];

            switch (mainCommand){
                case "quit":
                    break;

                case "read":
                    if (commandChain.length == 2)
                        loadNamespaces(commandChain[1]);
                    break;

                case "print":
                    showNamespaces();
                    break;

                default:
                    System.out.println("Unknown command: " + mainCommand);
                    printInteractiveUsage();
            }
        }
    }


    public static void main (String [] args)  {
        if (args.length == 0 || args[0].equals("-h") || args[0].equals("--help")) {
            printUsage();
            return;
        }

        String command = args[0];
        FileBasedDMS fileBasedDesignModelStorage = new FileBasedDMS();
        switch (command) {
            case "-i":
            case "--interactive":

                fileBasedDesignModelStorage.runInteractiveMode();
                break;
            case "-r":
            case "--read":
                if (args.length != 2) {
                    printUsage();
                }
                else {
                    fileBasedDesignModelStorage.loadNamespaces(args[1]);
                    fileBasedDesignModelStorage.showNamespaces();
                }
                break;
            default:
                System.out.println("Unknown command: " + command);
                printUsage();
        }
    }

    private static void printUsage() {
        System.out.println("Usage: java FileBasedDMS <command> [<args>]");
        System.out.println("Commands:");
        System.out.println(" -i, --interactive - switch to interactive mode");
        System.out.println(" -r, --read <directory> - read xxp files from a directory and prints its content on standard output");
        System.out.println("  -h, --help  - Show this help message");
    }
    private static void printInteractiveUsage() {
        System.out.println("Usage: <interactive commands> [<args>]");
        System.out.println("Commands:");
        System.out.println(" read <directory> - read the package from a directory of xxp files");
        System.out.println(" write <directory> - write the package to a directory (if it does not exist, it will be created)");
        System.out.println(" add workflow <name> - create a new workflow in the package");
        System.out.println(" add workflow <name> from <workflow_name> - create a new workflow from existing workflow");
        System.out.println(" add task <name> <workflow> - create a new task in the workflow");
        System.out.println(" print - prints the current package");
        System.out.println(" help - Show this help message");
        System.out.println(" quit - quit the application");



    }
}