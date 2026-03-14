using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

public class Keyword : RegexLexeme {
    public Keyword(string pattern, string groupName) : base(pattern, RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = groupName;
    }
}