import matplotlib.pyplot as plt
import os
import numpy as np

def plot_phonon_results(
   bs_energies,
   dos,
   dos_energies,
   all_k_point_distances,
   special_k_point_distances,
   special_k_point_labels,
   discontinuities,
   y_label,
   struct_formula,
   cif_filename_base,
   supercell_dims, 
   delta,
   fmax,
   output_dir
):
   """Generates and saves the phonon band structure and DOS plot."""
   plt.rc("figure", dpi=150)

   fig = plt.figure(1, figsize=(7, 4))
   bs_ax = fig.add_axes([0.12, 0.07, 0.67, 0.85])

   # Determine y‐axis limits
   bsmin = np.min(bs_energies) if bs_energies.size > 0 else 0
   bsmax = np.max(bs_energies) if bs_energies.size > 0 else 0
   y_range     = bsmax - bsmin
   extra_space = y_range * 0.20
   y_bottom    = bsmin - extra_space
   y_top       = bsmax + extra_space

   # Plot each phonon band
   for band in bs_energies.T:
      bs_ax.plot(all_k_point_distances, band, color='blue')

   # Vertical lines at each special k-point
   for dist in special_k_point_distances:
      bs_ax.axvline(dist, color='gray', linestyle='--')

   # —————————————————————————————
   # Merge labels at discontinuities:
   merged_distances = []
   merged_labels    = []
   skip_next = False

   for i, (dist, lbl) in enumerate(zip(special_k_point_distances, special_k_point_labels)):
      if skip_next:
         skip_next = False
         continue

      if i in discontinuities:
         # Merge label_i and label_{i+1}
         next_lbl = special_k_point_labels[i+1]
         merged_distances.append(dist)
         merged_labels.append(f"{lbl}|{next_lbl}")
         skip_next = True
      else:
         merged_distances.append(dist)
         merged_labels.append(lbl)

   bs_ax.set_xticks(merged_distances)
   bs_ax.set_xticklabels(merged_labels)
   # —————————————————————————————

   bs_ax.set_xlabel("Wave vector", fontsize=14)
   bs_ax.set_ylabel(y_label, fontsize=14)
   bs_ax.set_xlim(all_k_point_distances.min(), all_k_point_distances.max())
   bs_ax.set_ylim(y_bottom, y_top)

   # DOS inset
   dos_ax = fig.add_axes([0.8, 0.07, 0.17, 0.85])
   dos_ax.fill_between(
      dos.get_weights(),
      dos_energies,
      y2=0,
      color="grey",
      edgecolor="black",
      lw=1
   )
   dos_ax.set_ylim(y_bottom, y_top)
   dos_ax.set_yticks([])
   dos_ax.set_xticks([])
   dos_ax.set_xlabel("DOS", fontsize=14)

   # Title
   supercell_str_title = f"({supercell_dims[0]}x{supercell_dims[1]}x{supercell_dims[2]})" if isinstance(supercell_dims, (list, tuple)) else f"({supercell_dims}x{supercell_dims}x{supercell_dims})"  
   supercell_str_filename = f"{supercell_dims[0]}x{supercell_dims[1]}x{supercell_dims[2]}" if isinstance(supercell_dims, (list, tuple)) else f"{supercell_dims}"  

   fig.suptitle(  
   f"Phonon band structure and DOS of {struct_formula} "  
   f"({cif_filename_base}) with {supercell_str_title} supercell",  
   fontsize=12,  
   y=1.02,  
   )  

   png_path = os.path.join(  
   output_dir,  
   f"phonon_bs_dos_{cif_filename_base}_N{supercell_str_filename}_D{delta}_F{fmax}.png"  
   )  
   svg_path = os.path.join(  
   output_dir,  
   f"phonon_bs_dos_{cif_filename_base}_N{supercell_str_filename}_D{delta}_F{fmax}.svg"  
   )
   plt.savefig(png_path, dpi=300, bbox_inches="tight")
   plt.savefig(svg_path, bbox_inches="tight")
   plt.close(fig)

   print(f"Phonon band structure and DOS plot saved to {png_path} and {svg_path}")