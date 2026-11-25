if VERSION < v"1.12-"
    # AtomType has missing union, PeriodicTorsion has NTuple
    # LazyArtifacts is only loaded in MollyLuxExt
    Aqua.test_all(Molly; stale_deps=(ignore=[:LazyArtifacts],), unbound_args=false)
else
    Aqua.test_all(Molly; stale_deps=(ignore=[:LazyArtifacts],))
end

# Can be done with Aqua but we need to exclude the re-exported names
if VERSION >= v"1.11-"
    reexported = union(names(StaticArrays), names(Unitful))
    undoc = setdiff(Docs.undocumented_names(Molly), reexported)
    @test isempty(undoc)
end
