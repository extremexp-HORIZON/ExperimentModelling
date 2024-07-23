package eu.extremexp.simplelang.cmdline;

import com.beust.jcommander.Parameter;

public class Arguments {
    @Parameter(
            description = "File to parse",
            required = true
    )
    String file;

    @Parameter(
            names = {"-o", "--output"},
            description = "Output file",
            required = true
    )
    String output = null;

    @Parameter(
            names = {"-r", "--runDot"},
            description = "Run dot and generate SVG directly (otherwise the dot file is generated)"
    )
    boolean runDot = false;

    @Parameter(
            names = {"-h", "--help"},
            help = true,
            description = "Show help"
    )
    boolean help;

}
