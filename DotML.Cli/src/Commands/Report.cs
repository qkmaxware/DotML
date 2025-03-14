using System.Diagnostics;
using System.Drawing;
using System.Numerics;
using CommandLine;
using DotML.Cli.Visualizations;

namespace DotML.Cli.Commands;

[Verb("reports", HelpText = "View training session reports.")]
public class Report : BaseCommand {

    public override void Action(AppData appData) {
        if (!TryShowInExplorer(appData.TrainingDirectory)) {
            Console.WriteLine("Unable to open training reports folder.");
        } else {
            Console.WriteLine($"Training directory '{appData.TrainingDirectory.FullName}' opening in new window.");
        }
    }

    /// <summary>
    /// Tries to open the OS file explorer to the directory
    /// </summary>
    /// <param name="dir">directory to show in explorer</param>
    public static bool TryShowInExplorer(DirectoryInfo dir) {
        try {
            var process = new Process();
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.CreateNoWindow = true;

            if (System.Environment.OSVersion.Platform == PlatformID.Win32NT) {
                process.StartInfo.FileName = "explorer.exe";
                process.StartInfo.ArgumentList.Add(dir.FullName);
                return process.Start();
            } else {
                process.StartInfo.FileName = "mimeopen";
                process.StartInfo.ArgumentList.Add(dir.FullName);
                return process.Start();
            }
        } catch {
            return false;
        }
    }
}