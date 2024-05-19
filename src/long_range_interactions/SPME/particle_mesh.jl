function particle_mesh(spme::SPME, cuQ)
    # Q_inv = fft(cuQ)
    # Q_inv .*= BC_cuda
    # Q_inv = ifft!(Q_inv) # Q_conv_theta, but do in place

    # A = 332.0637132991921
    # E = 0.5 * A* sum(real(Q_inv) .* real(cuQ))
    # F = None
    # return E, F
end
