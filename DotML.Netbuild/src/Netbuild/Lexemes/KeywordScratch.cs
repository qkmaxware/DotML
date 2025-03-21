using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordScratch : RegexLexeme {
    public KeywordScratch() : base(@"\G\s*\b(?<value>SCRATCH)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}