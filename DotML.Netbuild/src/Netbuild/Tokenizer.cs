using System.Data;

namespace DotML.Network.IO.Netbuild;

internal class Tokenizer {
    
    private Lexeme[] all = [
        new KeywordFrom(),
        new KeywordScratch(),
        new KeywordInput(),
        new KeywordAdd(),
        new KeywordArg(),
        new KeywordName(),
        new KeywordRemove(),
        new KeywordReplace(),
        new KeywordWith(),
        new KeywordInsert(),
        new KeywordAfter(),
        new KeywordBefore(),
        new KeywordPretrain(),
        new KeywordAs(),

        new OperatorColon(),
        new OperatorAssign(),

        new Identifier(),
        new Number(),

        new Comment()
    ];
    
    public List<Token> GetTokens(string text) {
        var tokens = new List<Token>();
        var start_index = 0;
        while (start_index < text.Length) {
            bool was_matched = false;

            foreach (var lexeme in all) {
                var token = lexeme.GetNext(start_index, text);
                if (token is null)
                    continue;
                
                was_matched = true;
                start_index += token.Length;
                if (lexeme is not Comment)
                    tokens.Add(token);
            }
        
            if (!was_matched) {
                throw new SyntaxErrorException($"Invalid symbol/token '{text[start_index]}' at position {start_index}");
            }
        }

        return tokens;
    }
}