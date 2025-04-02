using System.Diagnostics;
using System.Drawing;
using System.Numerics;
using CommandLine;
using DotML.Cli.Visualizations;

namespace DotML.Cli.Commands;

[Verb("reports", HelpText = "View generated netflow reports.")]
public class Report : BaseCommand {

    public enum SubCommand {
        none,
        list,
        open,
        rm,
        clear
    }

    [Value(0, MetaName = "sub-command", HelpText = "Report action (list, open, rm, clear)", Required = false, Default = SubCommand.none)]
    public SubCommand Cmd {get; set;}

    [Value(1, MetaName = "name", HelpText = "Report name", Required = false)]
    public string? ReportName {get; set;}

    public override void Action(AppData appData) {
        switch (Cmd) {
            case SubCommand.list:
                list_all_reports(appData); break;
            case SubCommand.open:
                open_report_dir(appData); break;
            case SubCommand.rm:
                delete_report_dir(appData); break;
            case SubCommand.clear:
                delete_all_reports(appData); break;
            default:
                open_reports_dir(appData); break;
        }
        
    }

    private void open_reports_dir(AppData appData) {
        if (!TryShowInExplorer(appData.ReportDirectory)) {
            Console.WriteLine($"Unable to open training reports folder.");
            Console.WriteLine($"Enter '{appData.ReportDirectory.FullName}' into your file explorer.");
        } else {
            Console.WriteLine($"Report directory '{appData.ReportDirectory.FullName}' opening in new window.");
        }
    }

    private void open_report_dir(AppData appData) {
        var report_to_open = appData.EnumerateReports().Where(x => x.Name == ReportName).FirstOrDefault();
        if (report_to_open is null) {
            Console.WriteLine($"Report '{ReportName}' doesn't exist");
            Console.WriteLine();
            list_all_reports(appData);
        } else {
            if (!TryShowInExplorer(report_to_open.Directory)) {
                Console.WriteLine($"Unable to open training report folder.");
                Console.WriteLine($"Enter '{report_to_open.Directory.FullName}' into your file explorer.");
            } else {
                Console.WriteLine($"Report directory '{report_to_open.Directory.FullName}' opening in new window.");
            }
        }
    }

    private void delete_report_dir(AppData appData) {
        var report_to_open = appData.EnumerateReports().Where(x => x.Name == ReportName).FirstOrDefault();
        if (report_to_open is not null) {
            try {
                report_to_open.Delete();
                Console.WriteLine($"Report '{report_to_open.Name}' deleted successfully.");
            } catch {
                Console.WriteLine($"Failed to delete report '{report_to_open.Name}'.");
            }
        } 
    }

    private void delete_all_reports(AppData appData) {
        var report_to_open = appData.EnumerateReports();
        bool success = true;
        foreach (var report in report_to_open) {
            try {
                report.Delete();
            } catch {
                success = false;
            }
        }

        if (success) {
            Console.WriteLine($"All reports deleted successfully.");
        } else {
            TryShowInExplorer(appData.ReportDirectory);
            Console.WriteLine($"Failed to delete some reports, manual intervention may be required.");
            Console.WriteLine($"Report directory '{appData.ReportDirectory.FullName}' opening in new window.");
        }
    }

    private void list_all_reports(AppData appData) {
        var reports = appData.EnumerateReports();
        string[] columns = ["NAME", "TYPE", "CREATED", "MODIFIED"];
        int[] column_lengths = [36, 36, 16, 16];

        for (var col = 0; col < columns.Length; col++) {
            var name = columns[col];
            var len = column_lengths[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        foreach (var report in reports) {
            Console.Write(ColumnValue(report.Name, column_lengths[0]));
            Console.Write(' ');

            Console.Write(ColumnValue(report.GetType().Name, column_lengths[1]));
            Console.Write(' ');

            Console.Write(ColumnValue(report.Created.ToString("yyyy-MM-dd hh:mm"), column_lengths[2]));
            Console.Write(' ');

            Console.Write(ColumnValue(report.Modified.ToString("yyyy-MM-dd hh:mm"), column_lengths[3]));
            Console.WriteLine();
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

            if (OperatingSystem.IsWindows()) {
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