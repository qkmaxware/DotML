namespace Qkmaxware.Simple.CommandLineParser;

public interface ICliApp
{
    public void AddArgument(string str);
    public IEnumerable<Option> GetOptions();
}