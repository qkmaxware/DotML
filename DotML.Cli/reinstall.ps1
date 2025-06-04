dotnet tool uninstall --global netflow; 

dotnet pack DotML.Cli.csproj -c Release;

dotnet tool install --configfile none.nuget.config --global --add-source ./nupkg netflow