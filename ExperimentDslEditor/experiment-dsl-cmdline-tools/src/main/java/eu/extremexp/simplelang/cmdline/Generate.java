package eu.extremexp.simplelang.cmdline;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;
import eu.extremexp.simplelang.dot.DotGenerator;
import eu.extremexp.simplelang.elements.Workflow;
import eu.extremexp.simplelang.parser2.ParseException;
import eu.extremexp.simplelang.parser2.XXPParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class Generate {

    private static final Logger logger = LoggerFactory.getLogger(Generate.class);

    public static void main(String[] args) {
        Arguments arguments = new Arguments();
        JCommander jCommander = JCommander.newBuilder().addObject(arguments).build();
        try {
            jCommander.parse(args);
        } catch (ParameterException ex) {
            System.out.println(ex.getMessage());
            ex.usage();
            System.exit(1);
        }

        if (arguments.help) {
            jCommander.usage();
            System.exit(0);
        }

        try (InputStream is = new BufferedInputStream(new FileInputStream(arguments.file))) {
            Workflow workflow = XXPParser.parse(is);
            DotGenerator dotGenerator = new DotGenerator(workflow);
            List<String> dot = dotGenerator.getDot();
            if (arguments.runDot) {
                Path tmpDotFile = Files.createTempFile("extrmexp", ".dot");
                String tmpDotFileName = tmpDotFile.toAbsolutePath().toString();
                Files.write(tmpDotFile, dot);
                ProcessBuilder pb = new ProcessBuilder("dot", "-Tsvg", tmpDotFileName);
                pb.redirectOutput(new File(arguments.output));
                Process process = pb.start();
                try {
                    int result = process.waitFor();
                    if (result != 0) {
                        System.out.println("Generating SVG failed");
                    }
                } catch (InterruptedException ex) {
                    logger.warn("Exception when waiting for dot", ex);
                }
            } else {
                Files.write(Path.of(arguments.output), dot);
            }
        } catch (IOException ex) {
            System.out.println("Error when reading/writing a file");
            logger.error("Error when reading input file", ex);
        } catch (ParseException ex) {
            System.out.println("Language error");
            System.out.println(ex.getMessage());
            logger.error("Language error", ex);
        }
    }
}
