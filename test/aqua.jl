if VERSION < v"1.12-"
    # AtomType has missing union, PeriodicTorsion has NTuple
    Aqua.test_all(Molly; unbound_args=false)
else
    Aqua.test_all(Molly)
end
