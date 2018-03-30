# Analyse results

export
    writepdb

function writepdb(filepath::AbstractString, universe::Universe)
    open(filepath, "w") do output
        for (i, c) in enumerate(universe.coords)
            at = universe.molecule.atoms[i]
            at_rec = AtomRecord(
                false,
                i,
                at.name,
                ' ',
                at.resname,
                "A",
                at.resnum,
                ' ',
                [10*c.x, 10*c.y, 10*c.z],
                1.0,
                0.0,
                "  ",
                "  "
            )
            println(output, pdbline(at_rec))
        end
    end
end
