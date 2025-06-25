import matplotlib.pyplot as plt
import os
import numpy as np

def plot_phonon_results(bs_energies, dos, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, y_label, struct_formula, cif_filename_base, supercell_n, delta, fmax, output_dir):
   """Generates and saves the phonon band structure and DOS plot."""
   plt.rc("figure", dpi=150)

   fig = plt.figure(1, figsize=(7, 4))
   bs_ax = fig.add_axes([0.12, 0.07, 0.67, 0.85])

   bsmin = np.min(bs_energies) if bs_energies.size > 0 else 0
   bsmax = np.max(bs_energies) if bs_energies.size > 0 else 0

   y_range = bsmax - bsmin
   extra_space = y_range * 0.20
   y_bottom = bsmin - extra_space
   y_top = bsmax + extra_space

   for band in bs_energies.T:
      bs_ax.plot(all_k_point_distances, band, color='blue')

   for dist in special_k_point_distances:
      bs_ax.axvline(dist, color='gray', linestyle='--')

   bs_ax.set_xticks(special_k_point_distances)
   bs_ax.set_xticklabels(special_k_point_labels)
   bs_ax.set_xlabel("Wave vector", fontsize=14)
   bs_ax.set_ylabel(y_label, fontsize=14)

   bs_ax.set_xlim(all_k_point_distances.min(), all_k_point_distances.max())
   bs_ax.set_ylim(y_bottom, y_top)

   dos_ax = fig.add_axes([0.8, 0.07, 0.17, 0.85])
   dos_ax.fill_between(dos.get_weights(), dos_energies, y2=0, color="grey", edgecolor="black", lw=1)

   dos_ax.set_ylim(y_bottom, y_top)
   dos_ax.set_yticks([])
   dos_ax.set_xticks([])
   dos_ax.set_xlabel("DOS", fontsize=14)

   fig.suptitle(
      f"Phonon band structure and DOS of {struct_formula} ({cif_filename_base}) with ({supercell_n}, {supercell_n}, {supercell_n}) supercell",
      fontsize=12,
      y=1.02,
   )

   plot_filename_png = os.path.join(output_dir, f"phonon_bs_dos_{cif_filename_base}_N{supercell_n}_D{delta}_F{fmax}.png")
   plot_filename_svg = os.path.join(output_dir, f"phonon_bs_dos_{cif_filename_base}_N{supercell_n}_D{delta}_F{fmax}.svg")
   plt.savefig(plot_filename_png, dpi=300, bbox_inches="tight")
   plt.savefig(plot_filename_svg, bbox_inches="tight")
   plt.close(fig) # Close the figure to free up memory

   print(f"Phonon band structure and DOS plot saved to {plot_filename_png} and {plot_filename_svg}")

def save_raw_data(bs_energies, dos_energies, all_k_point_distances, special_k_point_distances, special_k_point_labels, supercell_n, delta, fmax, output_dir):
   """Saves raw band structure and DOS data to text files."""
   # Reshape bs_energies to 2D before saving
   # Assuming bs_energies shape is (num_k_points, num_bands, num_spin_channels)
   # We want to flatten the last dimension into the second, resulting in (num_k_points, num_bands * num_spin_channels)
   bs_energies_2d = bs_energies.reshape(bs_energies.shape[0], -1)
   np.savetxt(os.path.join(output_dir, f"band_structure_energies_N{supercell_n}_D{delta}_F{fmax}.txt"), bs_energies_2d)

   np.savetxt(os.path.join(output_dir, f"dos_energies_N{supercell_n}_D{delta}_F{fmax}.txt"), dos_energies)
   np.savetxt(os.path.join(output_dir, f"k_point_distances_N{supercell_n}_D{delta}_F{fmax}.txt"), all_k_point_distances)

   with open(os.path.join(output_dir, f"special_k_points_N{supercell_n}_D{delta}_F{fmax}.txt"), 'w') as f:
      f.write("Special K-point Distances:\n")
      for dist in special_k_point_distances:
         f.write(f"{dist}\n")
      f.write("\nSpecial K-point Labels:\n")
      for label in special_k_point_labels:
         f.write(f"{label}\n")

   print("Raw band structure and DOS data saved.")