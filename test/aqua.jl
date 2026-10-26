if VERSION < v"1.12-"
    # AtomType has missing union, PeriodicTorsion has NTuple
    Aqua.test_all(Molly; unbound_args=false)
else
    Aqua.test_all(Molly)
end

# Can be done with Aqua but we need to exclude the re-exported names
if VERSION >= v"1.11-"
    reexported = union(names(StaticArrays), names(Unitful))
    undoc = setdiff(Docs.undocumented_names(Molly), reexported)
    @test isempty(undoc)
end
