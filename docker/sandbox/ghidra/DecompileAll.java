// Headless Ghidra script: decompile every function of the loaded program
// to a single C file. Run via ghidra-decompile (analyzeHeadless -postScript).
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DecompileAll extends GhidraScript {

    @Override
    public void run() throws Exception {
        if (currentProgram == null) {
            println("DecompileAll: no program loaded");
            return;
        }

        String[] args = getScriptArgs();
        String outputPath = (args.length > 0) ? args[0] : "/tmp/decompiled.c";

        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);

        StringBuilder out = new StringBuilder();
        out.append("// program: ").append(currentProgram.getName()).append('\n');

        FunctionIterator functions = currentProgram.getListing().getFunctions(true);
        int ok = 0;
        int total = 0;
        while (functions.hasNext() && !monitor.isCancelled()) {
            Function function = functions.next();
            total++;
            DecompileResults result = decompiler.decompileFunction(function, 60, monitor);
            if (result != null && result.decompileCompleted() && result.getDecompiledFunction() != null) {
                out.append('\n').append(result.getDecompiledFunction().getC());
                ok++;
            } else {
                out.append("\n// failed to decompile: ").append(function.getName()).append('\n');
            }
        }
        decompiler.dispose();

        Files.write(Paths.get(outputPath), out.toString().getBytes(StandardCharsets.UTF_8));
        println("DecompileAll: wrote " + ok + "/" + total + " functions to " + outputPath);
    }
}
