# Check all force field XML files in openmm/openmmforcefields repositories read in okay

using Molly
using EzXML
using Suppressor

const suppress_warnings = true
const prefix = "."
const projects = ["openmm", "openmmforcefields"]

function read_all_xml()
    c_fine, c_err = 0, 0
    for project in projects
        for (d, _, fps) in walkdir(joinpath(prefix, project))
            for fp in fps
                if endswith(fp, ".xml")
                    jfp = joinpath(d, fp)
                    ff_xml = parsexml(read(jfp))
                    ff = root(ff_xml)
                    if ff.name != "ForceField"
                        continue
                    end
                    try
                        if suppress_warnings
                            @suppress_err MolecularForceField(jfp)
                        else
                            MolecularForceField(jfp)
                        end
                        c_fine += 1
                    catch e
                        c_err += 1
                        println(jfp, " - ", e)
                    end
                end
            end
        end
    end
    println()
    println("Total: ", c_fine + c_err)
    println("Fine: ", c_fine)
    println("Error: ", c_err)
end

read_all_xml()
