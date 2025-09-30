namespace Qkmaxware.Parsing
{
    public class Unit
    {
        private Unit() { }

        public static Unit Instance { get; private set; } = new Unit();
    }
}