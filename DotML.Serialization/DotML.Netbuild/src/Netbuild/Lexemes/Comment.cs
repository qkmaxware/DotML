using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

public class Comment : RegexLexeme {
    public Comment() : base(@"\G\s*#(?<value>[^\n]+)\s*", RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}