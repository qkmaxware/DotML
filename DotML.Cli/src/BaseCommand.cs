namespace DotML.Cli;

public abstract class BaseCommand {

    public static readonly int OKAY = 0;
    public static readonly int GENERIC_ERROR = 1;

    public abstract void Action(AppData appData);

    public int TryDoAction(AppData appData) {
        try {
            Console.WriteLine();
            Action(appData);
            Console.WriteLine();
            return OKAY;
        } 
        catch (ErrorCodeException ex) {
            return ex.ErrorCode;
        }
        catch (Exception e) {
            Console.WriteLine("An unexpected error occurred:");
            Console.WriteLine(e);
            return GENERIC_ERROR;
        }
    }

    protected string ColumnValue(object obj, int colLength) {
        var str = Truncate(obj.ToString() ?? string.Empty, colLength);
        return str.PadRight(colLength, ' ');
    }

    protected string Truncate(string str, int maxLength) {
        if (str.Length <= maxLength) {
            return str;
        }
        
        // IE "some long string" becomes "some long str..."
        return str.Substring(0, maxLength - 3) + "...";
    }
}

public class ErrorCodeException : System.Exception {
    public int ErrorCode {get; private set;}
    public ErrorCodeException(int error, string? message)  : base(message) {
        this.ErrorCode = error;
    }
    public ErrorCodeException(int error, string? message, Exception? inner) : base(message, inner) {}
}