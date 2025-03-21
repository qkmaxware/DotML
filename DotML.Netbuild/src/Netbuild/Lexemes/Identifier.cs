using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class Identifier : RegexLexeme {
    public Identifier() : base(@"\G\s*(?:(?<value>([a-zA-Z_][a-zA-Z_\-0-9]*))|""(?<value>(?:[^""\\]|\\.)*)""|'(?<value>(?:[^'\\]|\\.)*)')\s*", RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}