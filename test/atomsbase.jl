using AtomsBase
using AtomsBaseTesting


@testset "AbstractSystem -> Molly System" begin
    system = make_test_system().system;
    #Update values to be something that works with Molly
    system = AbstractSystem(system; 
                boundary_conditions = [Periodic(), Periodic(), Periodic()],
                bounding_box = [[1.54732, 0.0, 0.0],
                                [0.0, 1.4654985, 0.0],
                                [0.0, 0.0, 1.7928950]]u"Å");
    molly_sys = System(system)
    @test test_approx_eq(system, molly_sys; common_only = true)

end

