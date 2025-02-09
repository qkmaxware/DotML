using DotML.Network;
using DotML.Network.Training;

public interface IRetentionPolicy<T> {
    public void Backup(string name, T backup);
}