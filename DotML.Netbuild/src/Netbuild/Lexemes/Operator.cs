using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

public class Operator : RegexLexeme {
    public Operator(string pattern, string groupName) : base(pattern, RegexOptions.Compiled) {
        this.GroupName = groupName;
    }
}