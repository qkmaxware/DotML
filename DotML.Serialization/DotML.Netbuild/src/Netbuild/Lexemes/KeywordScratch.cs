using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordScratch : Keyword {
    public KeywordScratch() : base(@"\G\s*\b(?<value>SCRATCH)\b\s*", "value") { }
}