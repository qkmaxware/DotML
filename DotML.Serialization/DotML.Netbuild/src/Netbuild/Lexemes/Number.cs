using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

public class Number : RegexLexeme {
    public Number() : base(@"\G\s*(?<value>(?:\+|\-)?\d+(?:\.\d*)?)\s*", RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}