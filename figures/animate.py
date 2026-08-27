import matplotlib.pyplot as plt
from matplotlib import animation
import jax.numpy as jnp


def animate_states(result, name, z_level):
    # PLOTTING
    states = result.states
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Plot 1: Scalar field (e.g., density)
    cax0 = axs[0].imshow(
        states[0, 0, :, :, z_level].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title("Density")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Plot 2: Vector magnitude (e.g., velocity magnitude)
    cax1 = axs[1].imshow(
        jnp.sqrt(
            states[0, 1, :, :, z_level] ** 2
            + states[0, 2, :, :, z_level] ** 2
            + states[0, 3, :, :, z_level] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax1, ax=axs[1])
    axs[1].set_title("Velocity Magnitude")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # Plot 3: Optional third field, e.g., pressure or similar (reusing animate_vector logic)
    cax2 = axs[2].imshow(
        states[0, 4, :, :, z_level].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax2, ax=axs[2])
    axs[2].set_title("Pressure")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    cax3 = axs[3].imshow(
        jnp.sqrt(
            states[0, 5, :, :, z_level] ** 2
            + states[0, 6, :, :, z_level] ** 2
            + states[0, 7, :, :, z_level] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )

    fig.colorbar(cax3, ax=axs[3])
    axs[3].set_title("Magnetic field")
    axs[3].set_xlabel("x")
    axs[3].set_ylabel("y")

    # Update function for all three plots
    def animate_all(i):
        cax0.set_array(states[i, 0, :, :, z_level].T)
        cax1.set_array(
            jnp.sqrt(
                states[i, 1, :, :, z_level] ** 2
                + states[i, 2, :, :, z_level] ** 2
                + states[i, 3, :, :, z_level] ** 2
            ).T
        )
        cax2.set_array(states[i, 4, :, :, z_level].T)
        cax3.set_array(
            jnp.sqrt(
                states[i, 5, :, :, z_level] ** 2
                + states[i, 6, :, :, z_level] ** 2
                + states[i, 7, :, :, z_level] ** 2
            ).T
        )
        return cax0, cax1, cax2

    ani = animation.FuncAnimation(fig, animate_all, frames=states.shape[0], interval=50)

    ani.save("turbulence_sr/figures/" + name + ".gif")
    plt.show()
